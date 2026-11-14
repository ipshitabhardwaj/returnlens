"""Budget-aware router + workflow simulator. Never auto-rejects: outputs one of 4 routes."""

import numpy as np, pandas as pd

AUTO, ASK, REVIEW, INSPECT = "AUTO_APPROVE", "ASK_MORE_EVIDENCE", "HUMAN_REVIEW", "PHYSICAL_INSPECTION"


def risk_score(r_reuse, r_contra):  # noisy-OR of the two independent risk indicators
    return 1 - (1 - r_reuse) * (1 - r_contra)


def daily_capacity(fc_p50, fc_p90, rho=0.06, cmax=400, buffer=0.5):
    """Inspection slots for the day: share rho of forecast volume, reserve part of the P90 surge, cap at cmax."""
    planned = rho * (fc_p50 + buffer * max(fc_p90 - fc_p50, 0))
    return int(min(cmax, round(planned)))


def route_day(
    day_claims, capacity, tau_low=0.15, tau_hi=0.5, u_hi=0.6, surge_ratio=1.0, surge_limit=1.6
):
    """day_claims: DataFrame[risk, unc]. Returns Series of routes."""
    routes = pd.Series(AUTO, index=day_claims.index)
    routes[(day_claims.unc >= u_hi) & (day_claims.risk < tau_hi)] = (
        ASK  # poor evidence -> ask, don't accuse
    )
    routes[(day_claims.risk >= tau_low) & (routes == AUTO)] = REVIEW  # digital human review
    cand = day_claims[(day_claims.risk >= tau_hi)].sort_values("risk", ascending=False)
    if surge_ratio > surge_limit:  # P90/P50 very high: keep physical slots for top cases
        capacity = int(capacity * 0.7)
    routes[cand.index[:capacity]] = INSPECT
    routes[cand.index[capacity:]] = REVIEW  # overflow -> human review, not rejection
    return routes


def simulate(arrivals, fc, claims_pool, policy, p_abuse=0.04, surge_mult=1.6, seed=0, **kw):
    """arrivals: Series(date -> true #returns). fc: DataFrame(date -> p50, p90).
    claims_pool: DataFrame[risk, unc, abusive] taken from HELD-OUT test data."""
    rng = np.random.default_rng(seed)
    rows = []
    for d, n in arrivals.items():
        surge = n > 1.3 * arrivals.median()
        p = min(p_abuse * (surge_mult if surge else 1), 0.5)  # ASSUMPTION: abuse rises in sale surges
        ab = rng.random(int(n)) < p
        pool_ab, pool_ok = claims_pool[claims_pool.abusive == 1], claims_pool[claims_pool.abusive == 0]
        batch = pd.concat(
            [
                pool_ab.sample(ab.sum(), replace=True, random_state=int(rng.integers(1e9))),
                pool_ok.sample((~ab).sum(), replace=True, random_state=int(rng.integers(1e9))),
            ]
        ).reset_index(drop=True)
        f = fc.loc[d]
        cap = policy(f, n, **kw)
        routes = route_day(batch, cap, surge_ratio=f.p90 / max(f.p50, 1))
        insp = routes == INSPECT
        rows.append(
            dict(
                date=d,
                n=int(n),
                cap=cap,
                inspected=int(insp.sum()),
                abuse=int(batch.abusive.sum()),
                caught=int((insp & (batch.abusive == 1)).sum()),
                legit_insp=int((insp & (batch.abusive == 0)).sum()),
                overflow=int(max(0, (batch.risk >= 0.5).sum() - cap)),
            )
        )
    r = pd.DataFrame(rows)
    r["pol"] = policy.__name__
    return r


# --- policies: how many physical inspections to allow today -------------------------------------------
def static_cap(f, n, cap=60, **_):
    return cap  # fixed daily slots


def forecast_cap(f, n, rho=0.06, cmax=400, **_):
    return daily_capacity(f.p50, f.p90, rho=rho, cmax=cmax)


def match_static_budget(arrivals, fc, pool, target_total, lo=1, hi=800, **kw):
    """Bisect the static daily cap so total inspections ~= target_total (fair, equal-budget comparison)."""
    while lo < hi:
        mid = (lo + hi) // 2
        tot = simulate(arrivals, fc, pool, static_cap, cap=mid, **kw).inspected.sum()
        lo, hi = (mid + 1, hi) if tot < target_total else (lo, mid)
    return lo
