import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import datetime, timedelta

def get_factor_weights(
    factor: pd.DataFrame,
    ascending: bool = False,
    top_n: int = None,
    quantile: float = None,
    weight_method: str = "equal",
    ) -> pd.DataFrame:
    """
    factor: DataFrame with columns ['date','token_id','factor']
    weight_method: "equal" or "factor"
    """
    # --- Selection Logic ----------------------------------------------------
    if quantile is not None:
        # Quantile selection
        selected = (
            factor.groupby("date", group_keys=False)
            .apply(
                lambda x: x[x["factor"] <= x["factor"].quantile(quantile)]
                if ascending
                else x[x["factor"] >= x["factor"].quantile(quantile)]
            )
        )
    elif top_n is not None:
        # Top-N selection
        selected = (
            factor.groupby("date", group_keys=False)
            .apply(
                lambda x: x.nsmallest(top_n, "factor")
                if ascending
                else x.nlargest(top_n, "factor")
            )
        )
    else:
        raise ValueError("You must specify either quantile or top_n")

    # --- Weighting Logic ----------------------------------------------------
    if weight_method == "equal":
        # All selected assets get equal weight per date
        selected["weight"] = 1 / selected.groupby("date")["token_id"].transform("count")

    elif weight_method == "factor":
        # Weight proportional to factor magnitude (or raw factor)
        # Here we use abs() so larger magnitude gets larger weight
        selected["abs_factor"] = selected["factor"].abs()
        selected["weight"] = (
            selected["abs_factor"]
            / selected.groupby("date")["abs_factor"].transform("sum")
        )
        selected = selected.drop(columns=["abs_factor"])
    else:
        raise ValueError("weight_method must be 'equal' or 'factor'")

    # --- Reshape into date × token_id matrix and forward-fill ----------------
    weights = (
        selected.set_index(["date", "token_id"])
        .weight.unstack("token_id")
        .groupby(pd.Grouper(freq="H"))
        .ffill()
        .fillna(0)
    )

    # --- LAG to remove lookahead -------------------------------------------
    # signal at time t becomes tradable at time t+lag
    weights = weights.shift(1)

    # after shift, first rows become NaN -> flat portfolio
    weights = weights.fillna(0)

    # --- Re-normalize to sum to 1 per hour ---------------------------------
    row_sums = weights.sum(axis=1)
    row_sums[row_sums == 0] = 1
    weights = weights.div(row_sums, axis=0)

    # # --- Enforce minimum weight of 1% and round to 2 decimals --------------------
    # weights = weights.apply(clamp_and_round, axis=1)

    return weights

def clamp_and_round(row):
    row = row.copy()

    # Step 1: Set minimum weight = 1%
    row[row < 0.001] = 0.001

    # Step 2: Renormalize so weights sum to 1
    row = row / row.sum()

    # Step 3: Round to 2 decimals
    row = row.round(3)

    # Step 4: Fix rounding drift (ensure sum == 1)
    drift = 1 - row.sum()

    # Add/subtract drift from the largest weight
    if abs(drift) >= 0.005:
        # If drift is large, adjust the max weight
        max_idx = row.idxmax()
        row[max_idx] += drift
    else:
        # If drift is tiny, round again
        max_idx = row.idxmax()
        row[max_idx] += drift

    # Final rounding
    row = row.round(2)

    return row

def quantile_map(row, q=5):
    # percentile rank within the row (ignores NaNs)
    r = row.rank(method="first", pct=True)
    # map (0,1] → {1, ..., q}
    return np.ceil(r * q).astype("Int64")  # nullable int; NaNs stay NaN

def compute_weights(long_mask, short_mask, factor, prices, method="equal", N=63):
    """
    long_mask, short_mask : DataFrames with 1 for selected assets and 0 otherwise
    factor : factor DataFrame
    prices : price DataFrame
    method : str
        "equal", "factor", "inverse_factor", "volatility", "inverse_volatility"
    N : int
        Lookback for volatility weighting
    """

    # ---------------------------
    # 1) Compute raw scores
    # ---------------------------
    if method == "equal":
        long_scores = long_mask.copy()
        short_scores = short_mask.copy()

    elif method == "factor":
        long_scores = factor * long_mask
        short_scores = factor * short_mask

    elif method == "inverse_factor":
        invf = 1 / factor.replace(0, np.nan)
        long_scores = invf * long_mask
        short_scores = invf * short_mask

    elif method in ["volatility", "inverse_volatility"]:
        # Compute rolling annualized vol for every asset
        vols = prices.pct_change().rolling(N).std().shift() * np.sqrt(365)

        if method == "volatility":
            long_scores = vols * long_mask
            short_scores = vols * short_mask
        else:
            inv_vol = 1 / vols.replace(0, np.nan)
            long_scores = inv_vol * long_mask
            short_scores = inv_vol * short_mask

    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # ---------------------------
    # 2) Normalize weights
    # ---------------------------
    long_w = long_scores.div(long_scores.sum(axis=1), axis=0).fillna(0)
    short_w = short_scores.div(short_scores.sum(axis=1), axis=0).fillna(0)

    return long_w, short_w

def get_factor_ls_portfolio(prices, 
                            factor, 
                            flipped=False, 
                            bt_start=None, 
                            bt_end=None, 
                            fees=0.000, 
                            visualize_legs=False,
                            volume=None,
                            volume_threshold=None,
                            weight_method="equal",
                            vol_lookback=30,
                            mcap=None,
                            top=None):

    use_volume_filter = (volume is not None) and (volume_threshold is not None)
    if use_volume_filter:
        volume_ok = (volume >= volume_threshold).astype(float)
        volume_ok = volume_ok.reindex_like(factor).fillna(0)

    # Restrict calculations to top by mcap if requested
    use_top_mcap = (mcap is not None) and (top is not None)
    if use_top_mcap:
        mcap_tmp = mcap.reindex_like(factor)
        # Per-row: identify top N tokens, make a mask
        top_mask = mcap_tmp.apply(lambda row: row >= row.nlargest(top).min(), axis=1).astype(float)
        # Everything outside top N will be NaN for downstream ops
        factor_for_q = factor.where(top_mask == 1)
    else:
        top_mask = None
        factor_for_q = factor

    # Compute quantiles once; only among top if requested
    factor_quantiles = factor_for_q.apply(quantile_map, axis=1)

    # Make sure to set factor_quantiles to NaN for tokens not in top, if relevant
    if use_top_mcap:
        factor_quantiles = factor_quantiles.where(top_mask == 1)

    # ---------------------------
    # Define long and short masks
    # ---------------------------
    if flipped:
        long_mask  = (factor_quantiles == 1).astype(float)
        short_mask = (factor_quantiles == 5).astype(float)
    else:
        long_mask  = (factor_quantiles == 5).astype(float)
        short_mask = (factor_quantiles == 1).astype(float)

    if use_top_mcap:
        # Explicitly zero out any asset NOT in top n per day
        long_mask  = long_mask.where(top_mask == 1, 0)
        short_mask = short_mask.where(top_mask == 1, 0)
    if use_volume_filter:
        long_mask  *= volume_ok
        short_mask *= volume_ok

    # ---------------------------
    # Compute weights
    # ---------------------------
    long_weights, short_weights = compute_weights(
        long_mask, short_mask, factor, prices,
        method=weight_method, N=vol_lookback
    )

    # Apply top mcap mask to weights as well
    if use_top_mcap:
        long_weights = long_weights.where(top_mask == 1, 0)
        short_weights = short_weights.where(top_mask == 1, 0)

    # Combine L/S weights
    weights = long_weights.add(-short_weights, fill_value=0)
    weights = weights.div(weights.abs().sum(axis=1), axis=0).fillna(0)

    # Avoid lookahead bias
    weights = weights.shift()

    # Apply top mask after shift
    if use_top_mcap:
        weights = weights.where(top_mask == 1, 0)

    # ---------------------------
    # Run Backtest
    # ---------------------------
    pfo = vbt.Portfolio.from_orders(
        close=prices[weights.columns].loc[weights.index].replace(0, np.nan).ffill().loc[bt_start:bt_end],
        size=weights.loc[bt_start:bt_end],
        size_type='targetpercent',
        init_cash=100,
        cash_sharing=True,
        group_by=True,
        call_seq="auto",
        fees=fees,
        direction='both'
    )

    # ---------------------------
    # Visualize L/S legs (optional)
    # ---------------------------
    if visualize_legs:

        def leg_portfolio(w, label_color):
            pf = vbt.Portfolio.from_orders(
                close=prices[w.columns].loc[w.index].replace(0, np.nan).ffill().loc[bt_start:bt_end],
                size=w.loc[bt_start:bt_end],
                size_type='targetpercent',
                init_cash=100,
                cash_sharing=True,
                group_by=True,
                call_seq="auto",
                fees=fees,
                direction='both'
            )
            return pf

        long_leg = leg_portfolio(long_weights, "green")
        short_leg = leg_portfolio(-short_weights, "red")

        long_leg.value().plot(label='Long Leg', color='green')
        short_leg.value().plot(label='Short Leg', color='red')
        plt.legend()
        plt.show()

    return pfo, weights.loc[bt_start:bt_end], pfo.trades.records_readable
