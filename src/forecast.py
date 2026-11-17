# PDF ki Listing C-2 (forecast.py) poora yahan paste karo
import numpy as np, pandas as pd, holidays, lightgbm as lgb
# Hand-curated sale events (EDIT: fill real dates each year from news / seller-central pages)
SALES = [
    ("2022-09-23", "big_sale_festive"),
    ("2023-10-08", "big_sale_festive"),
    ("2024-09-27", "big_sale_festive"),
    ("2025-09-23", "big_sale_festive"),
    ("2026-10-01", "big_sale_festive"),
    ("2023-01-15", "republic_day_sale"),
    ("2024-01-14", "republic_day_sale"),
    ("2025-01-13", "republic_day_sale"),
    ("2026-01-12", "republic_day_sale"),
]
FESTIVALS = {"Diwali": "Diwali", "Holi": "Holi", "Dussehra": "Dussehra"}
def festival_table(years):
    rows = []
    for y in years:
        for d, n in holidays.India(years=y).items():
            for key in FESTIVALS:
                if key in n:
                    rows.append((pd.Timestamp(d), key))
    for d, n in SALES:
        rows.append((pd.Timestamp(d), n))
    return pd.DataFrame(rows, columns=["date", "event"]).drop_duplicates().sort_values("date")
def add_event_features(
    df, events, names=("Diwali", "Holi", "Dussehra", "big_sale_festive", "republic_day_sale")
):
    """Moving-holiday features: signed days to nearest occurrence of each event (clipped to +-45)."""
    df = df.copy()
    for ev in names:
        dates = events.loc[events.event == ev, "date"].values
        if len(dates) == 0:
            df[f"d2_{ev}"] = 45
            continue
        diff = (df["date"].values[:, None] - dates[None, :]).astype("timedelta64[D]").astype(int)
        nearest = diff[np.arange(len(df)), np.abs(diff).argmin(1)]
        df[f"d2_{ev}"] = np.clip(nearest, -45, 45)
    df["dow"] = df.date.dt.dayofweek
    df["month"] = df.date.dt.month
    df["dom"] = df.date.dt.day
    return df
def add_lags(df, col="y", lags=(7, 14, 28), horizon=1):
    df = df.copy()
    for L in lags:
        df[f"lag{L}"] = df[col].shift(L + horizon - 1)
    df["roll7"] = df[col].shift(horizon).rolling(7).mean()
    return df
QS = (0.1, 0.5, 0.9)
def fit_quantile(train, feats, target="y"):
    return {
        q: lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            verbose=-1,
        ).fit(train[feats], train[target])
        for q in QS
    }
def predict_quantile(models, df, feats):
    out = pd.DataFrame(
        {f"p{int(q*100)}": m.predict(df[feats]) for q, m in models.items()}, index=df.index
    )
    return out.apply(np.sort, axis=1, result_type="broadcast")  # fix quantile crossing
def pinball(y, q_pred, q):
    d = y - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))
def wql(y, P):  # weighted quantile loss over p10/p50/p90
    return float(sum(2 * pinball(y, P[f"p{int(q*100)}"], q) for q in QS) / (len(QS) * np.abs(y).mean()))
def conformal_widen(y_cal, P_cal, P_test, target=0.8):
    """Split-conformal fix for too-narrow p10/p90 bands: widen by the (target)-quantile of calibration misses."""
    s = np.maximum(P_cal.p10.values - y_cal, y_cal - P_cal.p90.values)
    k = min(1.0, target * (1 + 1 / len(s)))
    qhat = np.quantile(s, k)
    out = P_test.copy()
    out["p10"] = out.p10 - qhat
    out["p90"] = out.p90 + qhat
    out["p10"] = out.p10.clip(lower=0)
    return out
def rolling_origin(df, feats, start="2024-07-01", step_days=60, horizon_days=60):
    """Honest time-series CV: train on the past only, test on the next window, then roll forward."""
    res = []
    for cut in pd.date_range(
        start, df.date.max() - pd.Timedelta(days=horizon_days), freq=f"{step_days}D"
    ):
        tr = df[df.date < cut]
        te = df[(df.date >= cut) & (df.date < cut + pd.Timedelta(days=horizon_days))]
        P = predict_quantile(fit_quantile(tr, feats), te, feats)
        res.append(
            dict(
                cut=cut.date(),
                wql=wql(te.y.values, P),
                cover=float(((te.y >= P.p10) & (te.y <= P.p90)).mean()),
            )
        )
    return pd.DataFrame(res)
