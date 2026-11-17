# PDF ki Listing C-1 (retail_returns.py) poora yahan paste karo
def retail_daily(xlsx_path, country="United Kingdom"):
    df = pd.concat(
        pd.read_excel(xlsx_path, sheet_name=None).values()
    )  # file has 2 sheets (2009-10, 2010-11)
    df = df.rename(columns={"InvoiceNo": "Invoice"})  # OpenML copy uses InvoiceNo
    df["date"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()
    df = df[df.Country == country]
    is_ret = df["Invoice"].astype(str).str.startswith("C")
    sales = df[~is_ret & (df.Quantity > 0)].groupby("date").Quantity.sum()
    rets = -df[is_ret].groupby("date").Quantity.sum()
    out = pd.DataFrame({"sales": sales, "returns": rets}).asfreq("D").fillna(0)
    out["return_rate"] = out.returns / out.sales.clip(lower=1)
    return out  # NOTE: shop closed ~Christmas->New Year, so zeros there are 'closed', not 'no demand'
