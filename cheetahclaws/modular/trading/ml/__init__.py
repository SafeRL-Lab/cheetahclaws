"""ML stacker — combines LLM trading signals with classical quant factors.

Honest framing: this does NOT predict the market. It models *whether
the agent's recommendation is going to be right*, using the agent's
own historical record + classical features as input.

If the model says "this BUY is unlikely to outperform SPY", that's a
useful signal even when the agent is bullish — it means the agent's
specific blind spots (which the ML layer learns from past mistakes)
apply to this case.
"""
from . import features, stacker

__all__ = ["features", "stacker"]
