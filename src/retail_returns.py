"""Track C - daily sales + returns series from Online Retail II (invoices starting with 'C' = cancellations/returns)."""
import pandas as pd

# Order + cancellation of the SAME huge quantity within minutes (data-entry artefacts, found during C1 checks).
ARTEFACT_INVOICES = ["541431", "C541433", "581483", "C581484"]


def load_raw(xlsx_path):
    df = pd.concat(pd.read_excel(xlsx_path, sheet_name=None).values())  # 2 sheets (2009-10, 2010-11)
    return df.rename(columns={"InvoiceNo": "Invoice"})  # OpenML copy uses InvoiceNo


def daily_from_raw(df, country="United Kingdom", drop_invoices=()):
    df = df.copy()
    df["Invoice"] = df["Invoice"].astype(str)
    if len(drop_invoices):
        bad = df["Invoice"].isin([str(x) for x in drop_invoices]) & (df["Quantity"].abs() > 10000)
        df = df[~bad]
    df["date"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()
    df = df[df.Country == country]
    is_ret = df["Invoice"].str.startswith("C")
    sales = df[~is_ret & (df.Quantity > 0)].groupby("date").Quantity.sum()
    rets = -df[is_ret].groupby("date").Quantity.sum()
    out = pd.DataFrame({"sales": sales, "returns": rets}).asfreq("D").fillna(0)
    out["return_rate"] = out.returns / out.sales.clip(lower=1)
    return out  # NOTE: zeros = shop closed (Saturdays, Christmas break), not 'no demand'


def retail_daily(xlsx_path, country="United Kingdom", drop_invoices=()):
    return daily_from_raw(load_raw(xlsx_path), country, drop_invoices)
