"""Alternative-data fetchers for the /trading pipeline.

Three sources that LLM analysis can actually add value on (vs. classical
quant factors which are already priced in):

  - insider:  SEC EDGAR Form 4 filings (officer/director buys & sells)
  - sentiment: LLM-scored news headlines (uses cheetahclaws's existing model)
  - trends:   Google Trends search volume (requires `pytrends`)

All three soft-fail to empty strings if the data isn't available, so the
analyze pipeline keeps working when SEC is down or pytrends isn't
installed.
"""
from . import insider, sentiment, trends

__all__ = ["insider", "sentiment", "trends"]
