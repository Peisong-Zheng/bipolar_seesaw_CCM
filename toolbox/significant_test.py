# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pyEDM import CCM

# def ccm_significance_statistic(
#     df_sd, 
#     df_pre,
#     E=4, 
#     tau=8, 
#     fraction=0.1, 
#     n_ran=20, 
#     libSizes="100 200 300 400 500 600 700",
#     Tp=0,
#     sample=100,
#     showPlot=True
# ):
#     """
#     Perform a CCM significance test by:
#       1) Building a DataFrame with X, Y from df_sd and df_pre.
#       2) Running CCM on the real data.
#       3) Generating 'n_ran' surrogate versions of X (with random perturbations),
#          each time re-running CCM, storing results in ran_ccm_list_xy.
#       4) Optionally plotting real vs. surrogate cross mappings.

#     Parameters
#     ----------
#     df_sd : pd.DataFrame
#         DataFrame containing at least ["age"] and one data column for X.
#     df_pre : pd.DataFrame
#         DataFrame containing at least ["age"] and one data column for Y.
#     E : int
#         Embedding dimension (default=4).
#     tau : int
#         Time delay (default=8).
#     fraction : float
#         Fraction of random variation for surrogate (default=0.1).
#     n_ran : int
#         Number of surrogate draws (default=20).
#     libSizes : str or list
#         Library sizes for CCM (default="100 200 300 400 500 600 700").
#     sample : int
#         Number of bootstrap samples in each CCM call (default=100).
#     showPlot : bool
#         Whether to show the resulting figure (default=True).

#     Returns
#     -------
#     ccm_out : pd.DataFrame
#         CCM output for the real data, containing columns like ["LibSize", "X:Y", "Y:X"].
#     ran_ccm_list_xy : list
#         List of CCM outputs (DataFrames) from each of the n_ran surrogate runs.
#     """

#     def randomize_stadial(stadial_data, fraction=0.1, seed=None):
#         """
#         1) Multiply original data by (1 + random variation in [-fraction, fraction]).
#         2) Chop in half and rejoin (destroys original time ordering).
#         """
#         if seed is not None:
#             np.random.seed(seed)
        
#         N = len(stadial_data)
#         # 1. Vary by ± fraction
#         variation = 1.0 + fraction * (2*np.random.rand(N) - 1.0)
#         randomized = stadial_data * variation
        
#         # 2. Chop and rejoin
#         half = N // 2
#         randomized_swapped = np.concatenate([randomized[half:], randomized[:half]])
        
#         return randomized_swapped

#     # Build combined DataFrame: time, X, Y
#     # We use the second column in df_sd and df_pre as X and Y, respectively.
#     df = pd.DataFrame({
#         "Time": df_pre["age"],
#         "X":    df_sd[df_sd.columns[1]],
#         "Y":    df_pre[df_pre.columns[1]]
#     })

#     # Real-data CCM
#     ccm_out = CCM(
#         dataFrame   = df,
#         E           = E,
#         tau         = tau,
#         columns     = "X",   # predictor
#         target      = "Y",   # target
#         libSizes    = libSizes,
#         sample      = sample,
#         random      = True,
#         replacement = False,
#         Tp          = Tp
#     )

#     # Generate surrogate draws
#     ran_ccm_list_xy = []
#     for i in range(n_ran):
#         # 1) Generate random surrogate for X
#         X_ran = randomize_stadial(df["X"].values, fraction=fraction)
        
#         # 2) Create DataFrame with the same Y but newly randomized X
#         df_surr = pd.DataFrame({
#             "Time": df["Time"],
#             "X":    X_ran,
#             "Y":    df["Y"].values
#         })
        
#         # 3) Run CCM for X->Y on the surrogate data
#         out_xy = CCM(
#             dataFrame   = df_surr,
#             E           = E,
#             tau         = tau,
#             columns     = "X",
#             target      = "Y",
#             libSizes    = libSizes,
#             sample      = sample,
#             random      = True,
#             replacement = False,
#             Tp          = Tp
#         )
#         ran_ccm_list_xy.append(out_xy)

#     # Optionally plot results
#     if showPlot:
#         fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))

#         # Plot surrogates X->Y
#         for i, out_xy in enumerate(ran_ccm_list_xy):
#             label_xy = "Random (X->Y)" if i == 0 else None
#             ax1.plot(out_xy["LibSize"], out_xy["X:Y"], 
#                      color="lightcoral", alpha=0.2, label=label_xy)

#         # Overlay real X->Y
#         ax1.plot(ccm_out["LibSize"], ccm_out["X:Y"], "ro-", label="Real (X->Y)")
#         ax1.set_title("X → Y")
#         ax1.set_xlabel("Library Size")
#         ax1.set_ylabel("Prediction Skill (rho)")
#         ax1.legend()

#         # Plot surrogates Y->X
#         for i, out_xy in enumerate(ran_ccm_list_xy):
#             label_yx = "Random (Y->X)" if i == 0 else None
#             ax2.plot(out_xy["LibSize"], out_xy["Y:X"], 
#                      color="skyblue", alpha=0.2, label=label_yx)

#         # Overlay real Y->X
#         ax2.plot(ccm_out["LibSize"], ccm_out["Y:X"], "bo-", label="Real (Y->X)")
#         ax2.set_title("Y → X")
#         ax2.set_xlabel("Library Size")
#         ax2.set_ylabel("Prediction Skill (rho)")
#         ax2.legend()

#         plt.tight_layout()
#         plt.show()

#     return ccm_out, ran_ccm_list_xy





# def ccm_significance_test(ccm_mean, ensemble_ccm, uni_dir=False, if_plot=False):
#     """
#     Test whether the CCM result for the mean is significantly different from that of the shifted ensemble.
    
#     Parameters:
#       ccm_mean : pandas.DataFrame
#           CCM output for the mean data. Must contain columns "LibSize", "X:Y", and "Y:X".
#       ensemble_ccm : list of pandas.DataFrame
#           A list of CCM outputs for each ensemble member, with the same columns as ccm_mean.
          
#     Returns:
#       bool: True if the CCM using SAT to predict pre is significantly different 
#             (i.e. the mean value is outside the 5th-95th percentile of the ensemble) 
#             AND the CCM using pre to predict SAT is not significant (i.e. the mean falls 
#             within the ensemble range). Returns False otherwise.
#     """
#     # Use the maximum LibSize as the test point.
#     max_lib = ccm_mean["LibSize"].max()
    

#     mean_sat2pre = np.mean(ccm_mean['X:Y'])
#     mean_pre2sat = np.mean(ccm_mean['Y:X'])
    
#     # Gather ensemble values at the maximum LibSize.
#     ens_sat2pre = []
#     ens_pre2sat = []
#     for ens_df in ensemble_ccm:
#         try:
#             # val_sat2pre = ens_df.loc[ens_df["LibSize"] == max_lib, "X:Y"].values[0]
#             # val_pre2sat = ens_df.loc[ens_df["LibSize"] == max_lib, "Y:X"].values[0]
#             val_sat2pre = np.mean(ens_df['X:Y'])
#             val_pre2sat = np.mean(ens_df['Y:X'])
#             ens_sat2pre.append(val_sat2pre)
#             ens_pre2sat.append(val_pre2sat)
#         except Exception as e:
#             print(f"Error extracting ensemble data: {e}")
    
#     ens_sat2pre = np.array(ens_sat2pre)
#     ens_pre2sat = np.array(ens_pre2sat)

#     if if_plot:
#         # in case uni_dir is false plot figure with two subplots
    
#         if uni_dir:
#             # plot the histogram of the ensemble values and a vertical line for the mean
#             fig, ax = plt.subplots(1, 1, figsize=(6, 4),dpi=100)
#             ax.hist(ens_sat2pre, bins=20, density=True, color='lightcoral', alpha=0.5, label='Ensemble SAT->pre')
#             ax.axvline(mean_sat2pre, color='red', linestyle='--', label='Mean SAT->pre')
#             ax.set_title(r'$\hat{pre}|M_{sat}$')
#             # add x-axis label
#             ax.set_xlabel("Prediction Skill (ρ)")
#             ax.set_ylabel("Frequency")
#             ax.legend()
#             plt.show()
#         else:
#             # plot the histogram of the ensemble values and a vertical line for the mean
#             fig, axes = plt.subplots(1, 2, figsize=(12, 4),dpi=100)
#             ax1 = axes[0]
#             ax2 = axes[1]
#             ax1.hist(ens_sat2pre, bins=20, density=True, color='lightcoral', alpha=0.5, label='Ensemble SAT->pre')
#             ax1.axvline(mean_sat2pre, color='red', linestyle='--', label='Mean SAT->pre')
#             ax1.set_title(r'$\hat{pre}|M_{sat}$')
#             # add x-axis label
#             ax1.set_xlabel("Prediction Skill (ρ)")
#             ax1.set_ylabel("Frequency")
    

#             # ax1.legend()
#             ax2.hist(ens_pre2sat, bins=20, density=True, color='skyblue', alpha=0.5, label='Ensemble pre->SAT')
#             ax2.axvline(mean_pre2sat, color='blue', linestyle='--', label='Mean pre->SAT')
#             ax2.set_title(r'$\hat{sat}|M_{pre}$')
#             # add x-axis label
#             ax2.set_xlabel("Prediction Skill (ρ)")
#             ax2.set_ylabel("Frequency")


#         # ax2.legend()
#         plt.show()
    
#     # Compute the 5th and 95th percentiles of the ensemble distributions.
#     lower_sat2pre = np.percentile(ens_sat2pre, 5)
#     upper_sat2pre = np.percentile(ens_sat2pre, 95)
#     lower_pre2sat = np.percentile(ens_pre2sat, 5)
#     upper_pre2sat = np.percentile(ens_pre2sat, 95)
    
#     # Condition 1: Mean SAT->pre prediction (X:Y) is outside the ensemble range.
#     significant_sat2pre = (mean_sat2pre > upper_sat2pre)
    
#     # Condition 2: Mean pre->SAT prediction (Y:X) is within the ensemble range.
#     non_significant_pre2sat = (mean_pre2sat <= upper_pre2sat)
    
#     # return significant_sat2pre and non_significant_pre2sat
#     if uni_dir:
#         return significant_sat2pre
#     else:
#         return significant_sat2pre and non_significant_pre2sat
##################################################### The following is the new version


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyEDM import CCM

def ccm_significance_test_v2(
    df_sd, 
    df_pre,
    E=4, 
    tau=8, 
    n_ran=20, 
    libSizes="100 200 300 400 500 600 700",
    Tp=0,
    sample=100,
    uni_dir=False,
    showPlot=True
):
    """
    Perform a CCM significance test by:
      1) Building a DataFrame with X, Y from df_sd and df_pre.
      2) Running CCM on the real data.
      3) Generating 'n_ran' surrogate versions of X (with random perturbations),
         each time re-running CCM, storing results in ran_ccm_list_xy.
      4) Optionally plotting real vs. surrogate cross mappings.

    Parameters
    ----------
    df_sd : pd.DataFrame
        DataFrame containing at least ["age"] and one data column for X.
    df_pre : pd.DataFrame
        DataFrame containing at least ["age"] and one data column for Y.
    E : int
        Embedding dimension (default=4).
    tau : int
        Time delay (default=8).

    n_ran : int
        Number of surrogate draws (default=20).
    libSizes : str or list
        Library sizes for CCM (default="100 200 300 400 500 600 700").
    sample : int
        Number of bootstrap samples in each CCM call (default=100).
    showPlot : bool
        Whether to show the resulting figure (default=True).

    Returns
    -------
    ccm_out : pd.DataFrame
        CCM output for the real data, containing columns like ["LibSize", "X:Y", "Y:X"].
    ran_ccm_list_xy : list
        List of CCM outputs (DataFrames) from each of the n_ran surrogate runs.
    """

    # def randomize_stadial(stadial_data, seed=None):
    #     """
    #     1) Multiply original data by (1 + random variation in [-fraction, fraction]).
    #     2) Chop in half and rejoin (destroys original time ordering).
    #     """
    #     if seed is not None:
    #         np.random.seed(seed)
        
    #     # randomly select a break point between 2/10 and 8/10 of the data
    #     break_point = np.random.randint(len(stadial_data)*3//7, len(stadial_data)*4//7)
    #     randomized_swapped = np.concatenate([stadial_data[break_point:], stadial_data[:break_point]])
        
    #     return randomized_swapped

    def randomize_stadial(stadial_data, seed=None):
        """
        1) Multiply original data by (1 + random variation in [-fraction, fraction]).
        2) Chop in half and rejoin (destroys original time ordering).
        """
        if seed is not None:
            np.random.seed(seed)
        
        # randomly select a break point between 2/10 and 8/10 of the data
        break_point = np.random.randint(len(stadial_data)//5, len(stadial_data)*4//5)
        randomized_swapped = np.concatenate([stadial_data[break_point:], stadial_data[:break_point]])
        
        return randomized_swapped

    # Build combined DataFrame: time, X, Y
    # We use the second column in df_sd and df_pre as X and Y, respectively.
    df = pd.DataFrame({
        "Time": df_pre["age"],
        "X":    df_sd[df_sd.columns[1]],
        "Y":    df_pre[df_pre.columns[1]]
    })

    column_name=df_sd.columns[1]
    target_name=df_pre.columns[1]


    # Real-data CCM
    ccm_out = CCM(
        dataFrame   = df,
        E           = E,
        tau         = tau,
        columns     = "X",   # predictor
        target      = "Y",   # target
        libSizes    = libSizes,
        sample      = sample,
        random      = True,
        replacement = False,
        Tp          = Tp
    )

    # create an array to store the randomly generated time X time series
    ran_time_series = np.zeros((n_ran, len(df["X"])))
    # Generate surrogate draws
    ran_ccm_list_xy = []
    for i in range(n_ran):
        # 1) Generate random surrogate for X
        X_ran = randomize_stadial(df["X"].values)
        # add the randomized time series to the array
        ran_time_series[i] = X_ran

        
        # 2) Create DataFrame with the same Y but newly randomized X
        df_surr = pd.DataFrame({
            "Time": df["Time"],
            "X":    X_ran,
            "Y":    df["Y"].values
        })
        
        # 3) Run CCM for X->Y on the surrogate data
        out_xy = CCM(
            dataFrame   = df_surr,
            E           = E,
            tau         = tau,
            columns     = "X",
            target      = "Y",
            libSizes    = libSizes,
            sample      = sample,
            random      = True,
            replacement = False,
            Tp          = Tp
        )
        ran_ccm_list_xy.append(out_xy)

    if showPlot:
        # create a figure and plot the original time series and the randomized time series
        fig1, ax = plt.subplots(1, 1, figsize=(10, 3),dpi=100)
        ax.plot(df["Time"], df["X"], label=column_name)
        # ax.plot(df["Time"], df["Y"], label=target_name)
        # plot the randomized time series
        for i in range(n_ran):
            ax.plot(df["Time"], ran_time_series[i], color='grey', alpha=0.3)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        plt.show()

    # Optionally plot results
    if showPlot:

        fig, ax = plt.subplots(figsize=(4, 4))

        libsize = ran_ccm_list_xy[0]["LibSize"].values


        yx_surrogates = np.column_stack([out_xy["Y:X"].values for out_xy in ran_ccm_list_xy])
        yx_min = yx_surrogates.min(axis=1)
        yx_max = yx_surrogates.max(axis=1)

        xy_surrogates = np.column_stack([out_xy["X:Y"].values for out_xy in ran_ccm_list_xy])
        xy_min = xy_surrogates.min(axis=1)
        xy_max = xy_surrogates.max(axis=1)

        # Fill between min and max for X->Y
        ax.fill_between(libsize, xy_min, xy_max, color="r", alpha=0.2, label='', edgecolor='none')

        # Fill between min and max for Y->X
        ax.fill_between(libsize, yx_min, yx_max, color="b", alpha=0.2, label='', edgecolor='none')


        ax.plot(ccm_out["LibSize"], ccm_out["Y:X"], "bo-",
                label=fr"$\rho$ ($\hat{{{column_name}}}\mid M_{{{target_name}}}$)")

        ax.plot(ccm_out["LibSize"], ccm_out["X:Y"], "ro-",
                label=fr"$\rho$ ($\hat{{{target_name}}}\mid M_{{{column_name}}}$)")
        
        # set the xlim to match the range of the libsize
        ax.set_xlim([libsize[0], libsize[-1]])

        # set ylim to be -0.1 to 1.1
        ax.set_ylim([-0.15, 1.15])

        ax.set_xlabel("Library Size")
        ax.set_ylabel("Prediction Skill (rho)")
        ax.legend()
        plt.tight_layout()
        plt.show()



    test_result=ccm_significance_hist(ccm_out, ran_ccm_list_xy, uni_dir=uni_dir, column_name=column_name, target_name=target_name, if_plot=True)

    return ccm_out, ran_ccm_list_xy, test_result





# def ccm_significance_hist(ccm_mean, ensemble_ccm, uni_dir=False, column_name='sat', target_name='pre', if_plot=False):
#     """
#     Test whether the CCM result for the mean is significantly different from that of the shifted ensemble.
    
#     Parameters:
#       ccm_mean : pandas.DataFrame
#           CCM output for the mean data. Must contain columns "LibSize", "X:Y", and "Y:X".
#       ensemble_ccm : list of pandas.DataFrame
#           A list of CCM outputs for each ensemble member, with the same columns as ccm_mean.
          
#     Returns:
#       bool: True if the CCM using SAT to predict pre is significantly different 
#             (i.e. the mean value is outside the 5th-95th percentile of the ensemble) 
#             AND the CCM using pre to predict SAT is not significant (i.e. the mean falls 
#             within the ensemble range). Returns False otherwise.
#     """
#     # Use the maximum LibSize as the test point.
#     max_lib = ccm_mean["LibSize"].max()
    

#     mean_sat2pre = np.mean(ccm_mean['X:Y'])
#     mean_pre2sat = np.mean(ccm_mean['Y:X'])
    
#     # Gather ensemble values at the maximum LibSize.
#     ens_sat2pre = []
#     ens_pre2sat = []
#     for ens_df in ensemble_ccm:
#         try:
#             # val_sat2pre = ens_df.loc[ens_df["LibSize"] == max_lib, "X:Y"].values[0]
#             # val_pre2sat = ens_df.loc[ens_df["LibSize"] == max_lib, "Y:X"].values[0]
#             val_sat2pre = np.mean(ens_df['X:Y'])
#             val_pre2sat = np.mean(ens_df['Y:X'])
#             ens_sat2pre.append(val_sat2pre)
#             ens_pre2sat.append(val_pre2sat)
#         except Exception as e:
#             print(f"Error extracting ensemble data: {e}")
    
#     ens_sat2pre = np.array(ens_sat2pre)
#     ens_pre2sat = np.array(ens_pre2sat)

#     if if_plot:
#         # in case uni_dir is false plot figure with two subplots
    
#         if uni_dir:
#             # plot the histogram of the ensemble values and a vertical line for the mean
#             fig, ax = plt.subplots(1, 1, figsize=(6, 4),dpi=100)
#             ax.hist(ens_sat2pre, bins=20, density=True, color='lightcoral', alpha=0.5, label=fr"H0 $\rho$ ($\hat{{{target_name}}}\mid M_{{{column_name}}}$)")
#             ax.axvline(mean_sat2pre, color='red', linestyle='--', label=fr"Real $\rho$ ($\hat{{{target_name}}}\mid M_{{{column_name}}}$)")
#             # ax.set_title(r'$\hat{pre}|M_{sat}$')
#             # add x-axis label
#             ax.set_xlabel("Prediction Skill (ρ)")
#             ax.set_ylabel("Frequency")
#             ax.legend()
#             plt.show()
#         else:
#             # plot the histogram of the ensemble values and a vertical line for the mean
#             fig, axes = plt.subplots(1, 2, figsize=(12, 4),dpi=100)
#             ax1 = axes[0]
#             ax2 = axes[1]
#             ax1.hist(ens_sat2pre, bins=20, density=True, color='lightcoral', alpha=0.5, label=fr"H0 $\rho$ ($\hat{{{target_name}}}\mid M_{{{column_name}}}$)")
#             ax1.axvline(mean_sat2pre, color='red', linestyle='--', label=fr"Real $\rho$ ($\hat{{{target_name}}}\mid M_{{{column_name}}}$)")
#             # ax1.set_title(r'$\hat{pre}|M_{sat}$')
#             # add x-axis label
#             ax1.set_xlabel("Prediction Skill (ρ)")
#             ax1.set_ylabel("Frequency")
    

#             # ax1.legend()
#             ax2.hist(ens_pre2sat, bins=20, density=True, color='skyblue', alpha=0.5, label=fr"H0 $\rho$ ($\hat{{{column_name}}}\mid M_{{{target_name}}}$)")
#             ax2.axvline(mean_pre2sat, color='blue', linestyle='--', label=fr"Real $\rho$ ($\hat{{{column_name}}}\mid M_{{{target_name}}}$)")
#             # ax2.set_title(r'$\hat{sat}|M_{pre}$')
#             # add x-axis label
#             ax2.set_xlabel("Prediction Skill (ρ)")
#             ax2.set_ylabel("Frequency")


#         # ax2.legend()
#         plt.show()
    
#     # Compute the 5th and 95th percentiles of the ensemble distributions.
#     lower_sat2pre = np.percentile(ens_sat2pre, 5)
#     upper_sat2pre = np.percentile(ens_sat2pre, 95)
#     lower_pre2sat = np.percentile(ens_pre2sat, 5)
#     upper_pre2sat = np.percentile(ens_pre2sat, 95)
    
#     # Condition 1: Mean SAT->pre prediction (X:Y) is outside the ensemble range.
#     significant_sat2pre = (mean_sat2pre > upper_sat2pre)
    
#     # Condition 2: Mean pre->SAT prediction (Y:X) is within the ensemble range.
#     non_significant_pre2sat = (mean_pre2sat <= upper_pre2sat)
    
#     # return significant_sat2pre and non_significant_pre2sat
#     if uni_dir:
#         return significant_sat2pre
#     else:
#         return significant_sat2pre and non_significant_pre2sat

############################################################################################################



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyEDM import CCM
import random

def ccm_significance_test(
    df_sd, 
    df_pre,
    E=4, 
    tau=8, 
    libSizes="100 200 300 400 500 600 700",
    Tp=0,
    sample=100,
    n_ran=20, 
    df_stadial_ran=None,
    uni_dir=False,
    showPlot=True
):
    # check if df_stadial_ran is provided, if it is not, through an error
    if df_stadial_ran is None:
        raise ValueError("df_stadial_ran is not provided.")



    

    def randomize_stadial(ran_stadial_data,  seed=None):

        N=len(ran_stadial_data)
        # 2. Chop and rejoin
        half = int(N // 2)
        randomized_swapped = np.concatenate([ran_stadial_data[half:], ran_stadial_data[:half]])
        
        return randomized_swapped

    # Build combined DataFrame: time, X, Y
    # We use the second column in df_sd and df_pre as X and Y, respectively.
    df = pd.DataFrame({
        "Time": df_pre["age"],
        "X":    df_sd[df_sd.columns[1]],
        "Y":    df_pre[df_pre.columns[1]]
    })

    column_name=df_sd.columns[1]
    target_name=df_pre.columns[1]


    # Real-data CCM
    ccm_out = CCM(
        dataFrame   = df,
        E           = E,
        tau         = tau,
        columns     = "X",   # predictor
        target      = "Y",   # target
        libSizes    = libSizes,
        sample      = sample,
        random      = True,
        replacement = False,
        Tp          = Tp
    )

    # select N_ran random time series from the df_stadial_ran (it is a list with 500 time series)
    ran_time_series = random.sample(df_stadial_ran, n_ran)


    # Generate surrogate draws
    ran_ccm_list_xy = []
    for i in range(n_ran):
        # 1) Generate random surrogate for X
        X_ran = randomize_stadial(ran_time_series[i]["stadial_percent"].values)
        
        # 2) Create DataFrame with the same Y but newly randomized X
        df_surr = pd.DataFrame({
            "Time": df["Time"],
            "X":    X_ran,
            "Y":    df["Y"].values
        })
        
        # 3) Run CCM for X->Y on the surrogate data
        out_xy = CCM(
            dataFrame   = df_surr,
            E           = E,
            tau         = tau,
            columns     = "X",
            target      = "Y",
            libSizes    = libSizes,
            sample      = sample,
            random      = True,
            replacement = False,
            Tp          = Tp
        )
        ran_ccm_list_xy.append(out_xy)

    if showPlot:
        # create a figure and plot the original time series and the randomized time series
        fig1, ax = plt.subplots(1, 1, figsize=(10, 3),dpi=100)
        ax.plot(df["Time"], df["X"], label=column_name)
        # ax.plot(df["Time"], df["Y"], label=target_name)
        # plot the randomized time series
        for i in range(n_ran):
            ax.plot(ran_time_series[i]["age"].values, ran_time_series[i]["stadial_percent"].values, color='grey', alpha=0.3)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        plt.show()

    # Optionally plot results
    if showPlot:

        fig, ax = plt.subplots(figsize=(4, 4))

        libsize = ran_ccm_list_xy[0]["LibSize"].values


        yx_surrogates = np.column_stack([out_xy["Y:X"].values for out_xy in ran_ccm_list_xy])
        yx_min = yx_surrogates.min(axis=1)
        yx_max = yx_surrogates.max(axis=1)

        xy_surrogates = np.column_stack([out_xy["X:Y"].values for out_xy in ran_ccm_list_xy])
        xy_min = xy_surrogates.min(axis=1)
        xy_max = xy_surrogates.max(axis=1)

        # Fill between min and max for X->Y
        ax.fill_between(libsize, xy_min, xy_max, color="r", alpha=0.2, label='', edgecolor='none')

        # Fill between min and max for Y->X
        ax.fill_between(libsize, yx_min, yx_max, color="b", alpha=0.2, label='', edgecolor='none')


        ax.plot(ccm_out["LibSize"], ccm_out["Y:X"], "bo-",
                label=fr"$\rho$ ($\hat{{{column_name}}}\mid M_{{{target_name}}}$)")

        ax.plot(ccm_out["LibSize"], ccm_out["X:Y"], "ro-",
                label=fr"$\rho$ ($\hat{{{target_name}}}\mid M_{{{column_name}}}$)")
        
        # set the xlim to match the range of the libsize
        ax.set_xlim([libsize[0], libsize[-1]])

        # set ylim to be -0.1 to 1.1
        ax.set_ylim([-0.15, 1.15])

        ax.set_xlabel("Library Size")
        ax.set_ylabel("Prediction Skill (rho)")
        ax.legend()
        plt.tight_layout()
        plt.show()



    test_result=ccm_significance_hist(ccm_out, ran_ccm_list_xy, uni_dir=uni_dir, column_name=column_name, target_name=target_name, if_plot=True)

    return ccm_out, ran_ccm_list_xy, test_result





def ccm_significance_hist(ccm_mean, ensemble_ccm, uni_dir=False, column_name='sat', target_name='pre', if_plot=False):
    """
    Test whether the CCM result for the mean is significantly different from that of the shifted ensemble.
    
    Parameters:
      ccm_mean : pandas.DataFrame
          CCM output for the mean data. Must contain columns "LibSize", "X:Y", and "Y:X".
      ensemble_ccm : list of pandas.DataFrame
          A list of CCM outputs for each ensemble member, with the same columns as ccm_mean.
          
    Returns:
      bool: True if the CCM using SAT to predict pre is significantly different 
            (i.e. the mean value is outside the 5th-95th percentile of the ensemble) 
            AND the CCM using pre to predict SAT is not significant (i.e. the mean falls 
            within the ensemble range). Returns False otherwise.
    """
    # Use the maximum LibSize as the test point.
    max_lib = ccm_mean["LibSize"].max()
    

    mean_sat2pre = np.mean(ccm_mean['X:Y'])
    mean_pre2sat = np.mean(ccm_mean['Y:X'])
    
    # Gather ensemble values at the maximum LibSize.
    ens_sat2pre = []
    ens_pre2sat = []
    for ens_df in ensemble_ccm:
        try:
            # val_sat2pre = ens_df.loc[ens_df["LibSize"] == max_lib, "X:Y"].values[0]
            # val_pre2sat = ens_df.loc[ens_df["LibSize"] == max_lib, "Y:X"].values[0]
            val_sat2pre = np.mean(ens_df['X:Y'])
            val_pre2sat = np.mean(ens_df['Y:X'])
            ens_sat2pre.append(val_sat2pre)
            ens_pre2sat.append(val_pre2sat)
        except Exception as e:
            print(f"Error extracting ensemble data: {e}")
    
    ens_sat2pre = np.array(ens_sat2pre)
    ens_pre2sat = np.array(ens_pre2sat)

    if if_plot:
        # in case uni_dir is false plot figure with two subplots
    
        if uni_dir:
            # plot the histogram of the ensemble values and a vertical line for the mean
            fig, ax = plt.subplots(1, 1, figsize=(6, 4),dpi=100)
            ax.hist(ens_sat2pre, bins=20, density=True, color='lightcoral', alpha=0.5, label=fr"H0 $\rho$ ($\hat{{{target_name}}}\mid M_{{{column_name}}}$)")
            ax.axvline(mean_sat2pre, color='red', linestyle='--', label=fr"Real $\rho$ ($\hat{{{target_name}}}\mid M_{{{column_name}}}$)")
            # ax.set_title(r'$\hat{pre}|M_{sat}$')
            # add x-axis label
            ax.set_xlabel("Prediction Skill (ρ)")
            ax.set_ylabel("Frequency")
            ax.legend()
            plt.show()
        else:
            # plot the histogram of the ensemble values and a vertical line for the mean
            fig, axes = plt.subplots(1, 2, figsize=(12, 4),dpi=100)
            ax1 = axes[0]
            ax2 = axes[1]
            ax1.hist(ens_sat2pre, bins=20, density=True, color='lightcoral', alpha=0.5, label=fr"H0 $\rho$ ($\hat{{{target_name}}}\mid M_{{{column_name}}}$)")
            ax1.axvline(mean_sat2pre, color='red', linestyle='--', label=fr"Real $\rho$ ($\hat{{{target_name}}}\mid M_{{{column_name}}}$)")
            # ax1.set_title(r'$\hat{pre}|M_{sat}$')
            # add x-axis label
            ax1.set_xlabel("Prediction Skill (ρ)")
            ax1.set_ylabel("Frequency")
    

            # ax1.legend()
            ax2.hist(ens_pre2sat, bins=20, density=True, color='skyblue', alpha=0.5, label=fr"H0 $\rho$ ($\hat{{{column_name}}}\mid M_{{{target_name}}}$)")
            ax2.axvline(mean_pre2sat, color='blue', linestyle='--', label=fr"Real $\rho$ ($\hat{{{column_name}}}\mid M_{{{target_name}}}$)")
            # ax2.set_title(r'$\hat{sat}|M_{pre}$')
            # add x-axis label
            ax2.set_xlabel("Prediction Skill (ρ)")
            ax2.set_ylabel("Frequency")


        # ax2.legend()
        plt.show()
    
    # Compute the 5th and 95th percentiles of the ensemble distributions.
    lower_sat2pre = np.percentile(ens_sat2pre, 5)
    upper_sat2pre = np.percentile(ens_sat2pre, 95)
    lower_pre2sat = np.percentile(ens_pre2sat, 5)
    upper_pre2sat = np.percentile(ens_pre2sat, 95)
    
    # Condition 1: Mean SAT->pre prediction (X:Y) is outside the ensemble range.
    significant_sat2pre = (mean_sat2pre > upper_sat2pre)
    
    # Condition 2: Mean pre->SAT prediction (Y:X) is within the ensemble range.
    non_significant_pre2sat = (mean_pre2sat <= upper_pre2sat)
    
    # return significant_sat2pre and non_significant_pre2sat
    if uni_dir:
        return significant_sat2pre
    else:
        return significant_sat2pre and non_significant_pre2sat
