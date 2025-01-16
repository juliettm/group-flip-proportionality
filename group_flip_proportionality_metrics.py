import pandas as pd
import numpy as np


def get_total_flips(df):
    """
    Calculate the total number of flips in the dataset

    Args:
        df: DataFrame with 'flip_yp_corrected' column

    Returns:
        tuple: (number of flips, flip rate)
    """
    num_flips = (df['flip_yp_corrected'] == 1).sum()
    flip_rate = num_flips / len(df)
    return num_flips, flip_rate


def get_group_flips(df, group_value):
    """
    Calculate the number and percentage of flips for a specific protected attribute group

    Args:
        df: DataFrame with 'pa' and 'flip_yp_corrected' columns
        group_value: Value of protected attribute to analyze (0 or 1)

    Returns:
        tuple: (number of flips, flip rate for this group)
    """
    group_df = df[df['pa'] == group_value]
    flips = (group_df['flip_yp_corrected'] == 1).sum()
    total_in_group = len(group_df)
    flip_rate = (flips / total_in_group) if total_in_group > 0 else 0
    return flips, flip_rate


def get_all_group_flips(df):
    """
    Calculate flips for both protected attribute groups

    Args:
        df: DataFrame with 'pa' and 'flip_yp_corrected' columns

    Returns:
        dict: Dictionary containing flip metrics for both groups
    """
    metrics = {}
    for group in [0, 1]:
        flips, percentage = get_group_flips(df, group)
        metrics[f'group_{group}'] = {
            'num_flips': flips,
            'percentage': percentage
        }
    return metrics


# Directional Flip Ratio (DFR)
def directional_flip_ratio(df):
    """
    Calculate the directional flip ratio beneficial_flips/harmful_flips

    Args:
        df: DataFrame with 'pa' and 'flip_yp_corrected' columns

    Returns:
        tuple: (directional flip ratio, Short explanation of the value)
    """
    flips_0_to_1 = len(df[(df['y_pred'] == 0) & (df['y_corrected'] == 1)])
    flips_1_to_0 = len(df[(df['y_pred'] == 1) & (df['y_corrected'] == 0)])

    if flips_1_to_0 == 0:
        if flips_0_to_1 == 0:
            return 1.0, 'No flips in either direction'
        return np.inf, 'Only beneficial flips'

    if flips_0_to_1 == 0 and flips_1_to_0 > 0:
        return 0.0, 'Only harmful flips'
    else:
        return flips_0_to_1 / flips_1_to_0, 'Regular calculation'

def harmful_flip_proportion(df):
    """
    Calculate the harmful flip proportion: harmful_flips/total_flips

    Args:
        df: DataFrame with 'pa' and 'flip_yp_corrected' columns

    Returns:
        tuple: (harmful flip proportion, Short explanation of the value)
    """
    harmful_flips = len(df[(df['y_pred'] == 1) & (df['y_corrected'] == 0)])
    total_flips = len(df[df['y_pred'] != df['y_corrected']])
    # hfp = harmful_flips / total_flips if total_flips != 0 else np.inf
    if total_flips == 0:
        return 0.0, 'No flips present in the dataset'

    if harmful_flips == 0:
        return harmful_flips, 0.0, 'No harmful flips'
    else:
        return harmful_flips, harmful_flips / total_flips, 'Regular calculation'


# Flip Rate difference (FRD) or Harmful Flip Rate Difference (HFRD)
def flip_rate_difference(flip_rate_group_0, flip_rate_group_1):
    """
    Calculate the flip rate difference (FRD): difference in flip-rate between two groups

    Args:
        flip_rate_group_0: flip rate of the unprivileged group
        flip_rate_group_1: flip rate of the privileged group

    Returns:
        value: flip rate difference
    """
    return abs(flip_rate_group_1 - flip_rate_group_0)

# Disparity Index (DI) or Harmful Disparity Index (HDI)
def disparity_index(flip_rate_group_0, flip_rate_group_1):
    """
    Calculate the disparity index (DI): division of the maximun flip rate of the groups by the minimum

    Args:
        flip_rate_group_0: flip rate of the unprivileged group
        flip_rate_group_1: flip rate of the privileged group

    Returns:
        tuple: (disparity index, Short explanation of the value)
    """
    if min(flip_rate_group_0, flip_rate_group_1) == 0:
        if max(flip_rate_group_0, flip_rate_group_1) == 0:
            di_value = 1.0
            di_status = 'Both values are zero'
        else:
            di_value = np.inf
            di_status = 'One value is zero'
    else:
        di_value = max(flip_rate_group_0, flip_rate_group_1) / min(flip_rate_group_0, flip_rate_group_1)
        di_status = 'Regular calculation'
    return di_value, di_status

# Group-Wise Flip Disparity (GFD) -- Flip disparity
def flip_disparity(flip_rate_group_0, flip_rate_group_1, flip_rate):
    """
    Calculate the flip disparity: Absolute difference of the disparity of flips rates by groups over the overall flip rate

    Args:
        flip_rate_group_0: flip rate of the unprivileged group
        flip_rate_group_1: flip rate of the privileged group
        flip_rate: overall flip rate

    Returns:
        tuple: (flip disparity, Short explanation of the value)
    """
    if min(flip_rate_group_0, flip_rate_group_1) == 0:
        if max(flip_rate_group_0, flip_rate_group_1) == 0:
            fd_value = 1.0
            fd_status = 'Both values are zero'
        else:
            fd_value = np.inf
            fd_status = 'One value is zero'
    else:
        fd_value = max(flip_rate_group_0 / flip_rate, flip_rate_group_1 / flip_rate) - min(flip_rate_group_0 / flip_rate, flip_rate_group_1 / flip_rate)
        fd_status = 'Regular calculation'

    return fd_value, fd_status
    # (max(flip_rate_group_0 / flip_rate, flip_rate_group_1 / flip_rate) - min(flip_rate_group_0 / flip_rate, flip_rate_group_1 / flip_rate))

# Relative flip disparity (RFD) or Relative harmful flip disparity (RHFD)
def normalized_flip_disparity(flip_rate_group_0, flip_rate_group_1):
    """
    Calculate the Relative flip disparity: compute a normalized flip disparity (absolute difference of the flip rates
    over the sum of the flip rates)

    Args:
        flip_rate_group_0: flip rate of the unprivileged group
        flip_rate_group_1: flip rate of the privileged group

    Returns:
        tuple: (relative flip disparity, Short explanation of the value)
    """
    sum_rates = flip_rate_group_0 + flip_rate_group_1
    if sum_rates == 0:
        nfd_value = 0.0  # No disparity when both are zero
        nfd_status = 'Both zero'
    else:
        nfd_value = abs(flip_rate_group_0 - flip_rate_group_1) / sum_rates
        nfd_status = 'Regular calculation'
    return nfd_value, nfd_status


def group_wise_flip_proportionality_metrics(flip_rate_group_0, flip_rate_group_1, flip_rate):
    """
    Calculate the flip disparity: Absolute difference of the disparity of flips rates by groups over the overall flip rate

    Args:
        flip_rate_group_0: flip rate of the unprivileged group
        flip_rate_group_1: flip rate of the privileged group
        flip_rate: overall flip rate

    Returns:
        tuple: (flip disparity, Short explanation of the value)
    """
    frd = flip_rate_difference(flip_rate_group_0, flip_rate_group_1)
    di, di_desc = disparity_index(flip_rate_group_0, flip_rate_group_1)
    fd, fd_desc = flip_disparity(flip_rate_group_0, flip_rate_group_1, flip_rate)
    nfd, nfd_desc = normalized_flip_disparity(flip_rate_group_0, flip_rate_group_1)

    return {
        'frd': frd,
        'di': di,
        'di_desc': di_desc,
        'fd': fd,
        'fd_desc': fd_desc,
        'nfd': nfd,
        'nfd_desc': nfd_desc
    }