"""
Discovery layer — automatic candidate generation.

The base `/trading analyze` requires you to supply a symbol. This package
adds the missing piece: scan a universe and surface tickers worth
analysing, by source:

  - insider_cluster : SEC EDGAR Form 4 clusters (multiple recent filings)
  - earnings_beat   : recent positive-surprise reporters with continuation
  - sector_rotation : leading sector ETFs + their top holdings
  - momentum_quality: factor intersection (high momentum AND high quality)
  - anomaly         : unusual volume / price gaps / vol spikes

Each scanner returns `Discovery` rows (see types.py) so the orchestrator
can merge + dedupe across sources.
"""
from .types import Discovery
from . import insider_cluster, earnings_beat, sector_rotation, momentum_quality
from . import anomaly, orchestrator

__all__ = [
    "Discovery",
    "insider_cluster", "earnings_beat", "sector_rotation",
    "momentum_quality", "anomaly", "orchestrator",
]
