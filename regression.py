import pandas as pd
from typing import Tuple
import numpy as np
import math
def ic(signal: pd.Series, ret: pd.Series) -> float:
    """Calculate the IC using pure Pandas."""
    return signal.unstack().T.corrwith(ret.unstack().T).mean()
    # aligned = pd.concat([signal, ret], axis=1, keys=["signal", "ret"]).dropna()
    # return aligned["signal"].corr(aligned["ret"])

def calc_long_short_return(
    pred: pd.Series,
    label: pd.Series,
    date_col: str = "datetime",
    quantile: float = 0.2,
    dropna: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """Calculate long-short return.

    Note:
        `label` must be raw stock returns.

    Parameters
    ----------
    pred : pd.Series
        stock predictions
    label : pd.Series
        stock returns
    date_col : str
        datetime index name
    quantile : float
        long-short quantile

    Returns
    ----------
    long_short_r : pd.Series
        daily long-short returns
    long_avg_r : pd.Series
        daily long-average returns

    """
    df = pd.DataFrame({"pred": pred, "label": label})
    if dropna:
        df.dropna(inplace=True)
    group = df.groupby(level=date_col)

    def N(x):
        return int(len(x) * quantile)

    r_long = group.apply(
        lambda x: x.nlargest(N(x), columns="pred").label.mean()
    )
    r_short = group.apply(
        lambda x: x.nsmallest(N(x), columns="pred").label.mean()
    )
    r_avg = group.label.mean()
    return (r_long - r_short) / 2, r_avg


def calc_long_short_annual_return(
    pred: pd.Series,
    label: pd.Series,
    date_col: str = "datetime",
    quantile: float = 0.2,
    dropna: bool = False,
    risk_free_rate: float = 0.04,  # non-rist rate4%
) -> Tuple[pd.Series, pd.Series, float, float, float, float]:
    """Calculate long-short return (using compound returns) and Sharpe ratio.

    Parameters
    ----------
    pred : pd.Series
        stock predictions
    label : pd.Series
        stock returns (raw returns)
    date_col : str
        datetime index name
    quantile : float
        long-short quantile (e.g., top 20% and bottom 20%)
    risk_free_rate : float
        risk-free rate (default 5%)

    Returns
    ----------
    long_short_r : pd.Series
        daily long-short returns
    long_avg_r : pd.Series
        daily long-average returns
    long_short_ann_return : float
        annualized long-short return
    long_short_ann_sharpe : float
        annualized long-short Sharpe ratio
    long_avg_ann_return : float
        annualized long-average return
    long_avg_ann_sharpe : float
        annualized long-average Sharpe ratio
    """
    df = pd.DataFrame({"pred": pred, "label": label})
    if dropna:
        df.dropna(inplace=True)
    
    group = df.groupby(level=date_col)

    def N(x):
        return int(len(x) * quantile)

    # Calculate long and short portfolio daily returns
    r_long = group.apply(
        lambda x: x.nlargest(N(x), columns="pred").label.mean()
    )
    r_short = group.apply(
        lambda x: x.nsmallest(N(x), columns="pred").label.mean()
    )
    # 对于 r_long
    # r_long = group.apply(
    #     lambda x: x.nlargest(N(x), columns="pred").label.mean() if x["pred"].nunique() > N(x) 
    #     else x.sample(N(x), random_state=10).label.mean()
    # )

    # 对于 r_short
    # r_short = group.apply(
    #     lambda x: x.nsmallest(N(x), columns="pred").label.mean() if x["pred"].nunique() > N(x) 
    #     else x.sample(N(x), random_state=10).label.mean()
    # )
    r_avg = group.label.mean()

    # Calculate daily long-short returns
    long_short_r = (r_long - r_short) / 2
    long_avg_r = r_avg

    # Calculate annualized returns (compound return method)
    def annualized_return(daily_returns):

        cumulative_return = (1 + daily_returns - 0.0004).prod() - 1 # considering transaction cost and price impact
        # cumulative_return = (1 + daily_returns).prod() - 1
        annualized_return = (1 + cumulative_return) ** (1/2) - 1
        return annualized_return

    long_short_ann_return = annualized_return(long_short_r)
    long_avg_ann_return = annualized_return(long_avg_r)

    # Calculate daily volatilities (standard deviation) and annualized volatility
    def annualized_volatility(daily_returns):
        return daily_returns.std() * np.sqrt(252)

    long_short_ann_volatility = annualized_volatility(long_short_r)
    long_avg_ann_volatility = annualized_volatility(long_avg_r)

    # Calculate Sharpe ratio: (Annualized return - Risk-Free Rate) / Annualized Volatility
    def sharpe_ratio(annualized_return, annualized_volatility, risk_free_rate):
        return (annualized_return - risk_free_rate) / annualized_volatility
        # return (annualized_return) / annualized_volatility
        

    long_short_ann_sharpe = sharpe_ratio(long_short_ann_return, long_short_ann_volatility, risk_free_rate)
    long_avg_ann_sharpe = sharpe_ratio(long_avg_ann_return, long_avg_ann_volatility, risk_free_rate)

    return long_short_r, long_avg_r, long_short_ann_return, long_short_ann_sharpe, long_avg_ann_return, long_avg_ann_sharpe




def calc_ic(pred: pd.Series, label: pd.Series, date_col="datetime", dropna=False):
        df = pd.DataFrame({"pred": pred, "label": label})
        ic = df.groupby(date_col).apply(lambda df: df["pred"].corr(df["label"]))
        ric = df.groupby(date_col).apply(
            lambda df: df["pred"].corr(df["label"], method="spearman")
        )
        if dropna:
            return ic.dropna(), ric.dropna()
        else:
            return ic, ric

def mae(
    signal: pd.Series, ret: pd.Series, dimension: str = "instrument"
) -> float:
    """Calculate the MAE of the signal.

    Parameters
    ----------
    signal : pd.Series
        The alpha signal organized as multi-indexed series (datetime, instrument).
    ret : pd.Series
        The return series organized as multi-indexed series (datetime, instrument).
    dimension: str
        The dimension to compute the metric along first.

    Returns
    -------
    float
        The MAE of the signal.

    """
    if dimension not in ["datetime", "instrument"]:
        raise ValueError(
            "Invalid dimension. Expected 'datetime' or 'instrument'."
        )

    # Align signal and ret series and drop NaNs
    signal, ret = signal.align(ret, join="inner")
    signal_clean = signal.dropna()
    ret_clean = ret.dropna()

    if dimension == "datetime":
        # Swap levels to have 'datetime' as the inner index
        signal_clean = signal_clean.swaplevel()
        ret_clean = ret_clean.swaplevel()

    # Calculate the MAE
    mae = abs(signal_clean - ret_clean).groupby(level=0).mean().mean()
    return mae


def mse(
    signal: pd.Series, ret: pd.Series, dimension: str = "instrument"
) -> float:
    if dimension not in ["datetime", "instrument"]:
        raise ValueError(
            "Invalid dimension. Expected 'datetime' or 'instrument'."
        )

    signal, ret = signal.align(ret, join="inner")
    signal_clean = signal.dropna()
    ret_clean = ret.dropna()

    if dimension == "datetime":
        signal_clean = signal_clean.swaplevel()
        ret_clean = ret_clean.swaplevel()

    mse = ((signal_clean - ret_clean) ** 2).groupby(level=0).mean().mean()
    return mse


def rmse(
    signal: pd.Series, ret: pd.Series, dimension: str = "instrument"
) -> float:
    """Calculate the RMSE of the signal.

    Parameters
    ----------
    signal : pd.Series
        The alpha signal organized as multi-indexed series (datetime, instrument).
    ret : pd.Series
        The return series organized as multi-indexed series (datetime, instrument).
    dimension: str
        The dimension to compute the metric along first.

    Returns
    -------
    float
        The RMSE of the signal.

    """
    import numpy as np

    if dimension not in ["datetime", "instrument"]:
        raise ValueError(
            "Invalid dimension. Expected 'datetime' or 'instrument'."
        )

    # Align signal and ret series and drop NaNs
    signal, ret = signal.align(ret, join="inner")
    signal_clean = signal.dropna()
    ret_clean = ret.dropna()

    if dimension == "datetime":
        # Swap levels to have 'datetime' as the inner index
        signal_clean = signal_clean.swaplevel()
        ret_clean = ret_clean.swaplevel()

    # Calculate the RMSE
    rmse = np.sqrt(
        ((signal_clean - ret_clean) ** 2).groupby(level=0).mean().mean()
    )
    return rmse


def mape(
    signal: pd.Series, ret: pd.Series, dimension: str = "instrument"
) -> float:
    if dimension not in ["datetime", "instrument"]:
        raise ValueError(
            "Invalid dimension. Expected 'datetime' or 'instrument'."
        )

    signal, ret = signal.align(ret, join="inner")
    signal_clean = signal.dropna()
    ret_clean = ret.dropna()

    if dimension == "datetime":
        signal_clean = signal_clean.swaplevel()
        ret_clean = ret_clean.swaplevel()

    mape = (
        (abs((signal_clean - ret_clean) / ret_clean))
        .groupby(level=0)
        .mean()
        .mean()
    )
    return mape


def mda(
    signal: pd.Series, ret: pd.Series, dimension: str = "instrument"
) -> float:
    if dimension not in ["datetime", "instrument"]:
        raise ValueError(
            "Invalid dimension. Expected 'datetime' or 'instrument'."
        )

    signal, ret = signal.align(ret, join="inner")
    signal_clean = signal.dropna()
    ret_clean = ret.dropna()

    if dimension == "datetime":
        signal_clean = signal_clean.swaplevel()
        ret_clean = ret_clean.swaplevel()

    signal_diff = signal_clean.diff()
    ret_diff = ret_clean.diff()

    mda = ((signal_diff * ret_diff) > 0).groupby(level=0).mean().mean()
    return mda


def r2_score(
    signal: pd.Series, ret: pd.Series, dimension: str = "instrument"
) -> float:
    """Calculate the R-squared of the signal.

    Parameters
    ----------
    signal : pd.Series
        The alpha signal organized as multi-indexed series (datetime, instrument).
    ret : pd.Series
        The return series organized as multi-indexed series (datetime, instrument).
    dimension: str
        The dimension to compute the metric along first.

    Returns
    -------
    float
        The R-squared of the signal.

    """
    if dimension not in ["datetime", "instrument"]:
        raise ValueError(
            "Invalid dimension. Expected 'datetime' or 'instrument'."
        )

    # Align signal and ret series and drop NaNs
    signal, ret = signal.align(ret, join="inner")
    signal_clean = signal.dropna()
    ret_clean = ret.dropna()

    if dimension == "datetime":
        # Swap levels to have 'datetime' as the inner index
        signal_clean = signal_clean.swaplevel()
        ret_clean = ret_clean.swaplevel()

    # Calculate the R-squared
    ss_res = ((signal_clean - ret_clean) ** 2).groupby(level=0).sum()
    ss_tot = (
        ((signal_clean - signal_clean.groupby(level=0).mean()) ** 2)
        .groupby(level=0)
        .sum()
    )
    r2 = 1 - ss_res.sum() / ss_tot.sum()
    return r2
