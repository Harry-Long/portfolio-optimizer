# mod/risk_tools.py
import numpy as np, pandas as pd

def to_simple_returns(prices: pd.DataFrame)->pd.DataFrame:
    return np.exp(np.log(prices).diff())-1.0

def portfolio_returns(ret: pd.DataFrame, w: pd.Series)->pd.Series:
    w=w.reindex(ret.columns).fillna(0.0); return (ret*w).sum(axis=1)

def portfolio_nav(p: pd.Series, s: float=1.0)->pd.Series:
    return s*(1+p).cumprod()

def annualized_cov(r: pd.DataFrame, periods_per_year=252)->pd.DataFrame:
    return r.cov()*periods_per_year

def risk_contribution(w: pd.Series, cov: pd.DataFrame)->pd.Series:
    wv=w.reindex(cov.index).fillna(0.0); tot=float(np.sqrt(max(wv.values@cov.values@wv.values,0.0)))
    if tot==0: return wv*0
    mrc=cov@wv; return wv*mrc/tot

def corr_matrix(r: pd.DataFrame)->pd.DataFrame:
    return r.corr()

def var_es_hist(s: pd.Series, alpha=0.95)->dict:
    q=s.quantile(1-alpha); es=s[s<=q].mean(); return {"VaR":float(q), "ES":float(es)}
