"""
Bull/Bear researcher agents for debate-based investment analysis.

Inspired by TradingAgents' debate pattern:
  1. Bull researcher builds a case for buying
  2. Bear researcher builds a case for selling
  3. They debate back and forth (configurable rounds)
  4. A research judge synthesises the debate into a recommendation

Each researcher uses BM25 memory to reference past similar situations.
"""
from __future__ import annotations

from .memory import get_memory


# ── Bull Researcher ────────────────────────────────────────────────────────

def get_bull_prompt(
    symbol: str,
    trade_date: str,
    reports: dict[str, str],
    bear_arguments: str = "",
    debate_round: int = 1,
) -> str:
    """Generate the bull researcher prompt.

    Args:
        symbol: instrument to analyze
        trade_date: current date
        reports: {"technical": ..., "fundamental": ..., "news": ...}
        bear_arguments: previous bear arguments to counter (empty for round 1)
        debate_round: current debate round
    """
    memory = get_memory("bull_researcher")
    past_memories = memory.get_memories(
        f"{symbol} {reports.get('technical', '')[:200]}",
        n_matches=3, symbol=symbol,
    )

    memory_section = ""
    if past_memories:
        memory_section = "\n## Past Similar Situations (from memory)\n"
        for m in past_memories:
            memory_section += f"\n**Situation**: {m['situation'][:200]}\n"
            memory_section += f"**Recommendation**: {m['recommendation'][:200]}\n"
            memory_section += f"**Outcome**: {m['outcome'][:100]}\n"
            memory_section += f"**Similarity**: {m['similarity']}\n"

    counter_section = ""
    if bear_arguments:
        counter_section = f"""
## Bear Analyst's Arguments (Round {debate_round - 1})
{bear_arguments}

You MUST address and counter each of the bear analyst's key points with specific data.
"""

    return f"""You are a senior BULL analyst at a top-tier investment firm, specializing in identifying growth opportunities and building bullish investment cases.

**Date**: {trade_date}
**Instrument**: {symbol}
**Debate Round**: {debate_round}

## Analyst Reports

### Technical Analysis
{reports.get('technical', 'Not available')}

### Fundamental Analysis
{reports.get('fundamental', 'Not available')}

### News & Sentiment
{reports.get('news', 'Not available')}
{memory_section}
{counter_section}

## Your Task

Build a compelling BULLISH case for {symbol}. You must:

1. **Growth Catalysts**: Identify specific growth drivers and competitive advantages
2. **Technical Support**: Reference bullish technical signals (support levels, momentum, trend)
3. **Fundamental Strengths**: Highlight strong financials, valuation upside, improving metrics
4. **Positive Sentiment**: Note any positive news catalysts or sentiment shifts
5. **Risk Mitigation**: Acknowledge risks but explain why they are manageable
6. **Price Target**: Suggest an upside target with reasoning

Be specific. Cite numbers from the reports. This is a professional debate — vague optimism is not convincing.

Format your response as a structured argument with clear sections.
End with: **BULL VERDICT**: [Strong Buy / Buy / Lean Buy] — [one-sentence thesis]
"""


# ── Bear Researcher ────────────────────────────────────────────────────────

def get_bear_prompt(
    symbol: str,
    trade_date: str,
    reports: dict[str, str],
    bull_arguments: str = "",
    debate_round: int = 1,
) -> str:
    """Generate the bear researcher prompt."""
    memory = get_memory("bear_researcher")
    past_memories = memory.get_memories(
        f"{symbol} {reports.get('technical', '')[:200]}",
        n_matches=3, symbol=symbol,
    )

    memory_section = ""
    if past_memories:
        memory_section = "\n## Past Similar Situations (from memory)\n"
        for m in past_memories:
            memory_section += f"\n**Situation**: {m['situation'][:200]}\n"
            memory_section += f"**Recommendation**: {m['recommendation'][:200]}\n"
            memory_section += f"**Outcome**: {m['outcome'][:100]}\n"

    counter_section = ""
    if bull_arguments:
        counter_section = f"""
## Bull Analyst's Arguments (Round {debate_round - 1})
{bull_arguments}

You MUST challenge each of the bull analyst's key points with specific counter-evidence.
"""

    return f"""You are a senior BEAR analyst at a top-tier investment firm, specializing in identifying risks, overvaluations, and building bearish cases.

**Date**: {trade_date}
**Instrument**: {symbol}
**Debate Round**: {debate_round}

## Analyst Reports

### Technical Analysis
{reports.get('technical', 'Not available')}

### Fundamental Analysis
{reports.get('fundamental', 'Not available')}

### News & Sentiment
{reports.get('news', 'Not available')}
{memory_section}
{counter_section}

## Your Task

Build a compelling BEARISH case for {symbol}. You must:

1. **Risk Factors**: Identify specific downside risks and vulnerabilities
2. **Technical Weakness**: Reference bearish technical signals (resistance, divergences, breakdown)
3. **Fundamental Concerns**: Highlight valuation issues, deteriorating metrics, debt concerns
4. **Negative Catalysts**: Note any negative news, regulatory risks, competitive threats
5. **Bull Trap Warnings**: Explain why bullish signals might be misleading
6. **Downside Target**: Suggest a downside target with reasoning

Be specific. Cite numbers from the reports. Challenge the bull case rigorously.

Format your response as a structured argument with clear sections.
End with: **BEAR VERDICT**: [Strong Sell / Sell / Lean Sell] — [one-sentence thesis]
"""


# ── Research Judge ─────────────────────────────────────────────────────────

def get_research_judge_prompt(
    symbol: str,
    trade_date: str,
    reports: dict[str, str],
    bull_case: str,
    bear_case: str,
    debate_rounds: int,
) -> str:
    """Generate the research judge prompt to synthesize the debate."""
    return f"""You are the Chief Research Officer at a leading investment firm. Your role is to objectively evaluate the bull and bear cases and deliver a decisive recommendation.

**Date**: {trade_date}
**Instrument**: {symbol}
**Debate Rounds Completed**: {debate_rounds}

## Analyst Reports Summary

### Technical Analysis
{reports.get('technical', 'Not available')[:500]}

### Fundamental Analysis
{reports.get('fundamental', 'Not available')[:500]}

## The Debate

### Bull Case
{bull_case}

### Bear Case
{bear_case}

## Your Task

1. **Evaluate Both Cases**: Which side presented stronger, data-backed arguments?
2. **Identify Consensus Points**: Where do both sides agree?
3. **Weight the Evidence**: Which data points are most reliable and actionable?
4. **Make a DECISIVE Call**: You must choose a side. "Hold" is only acceptable when evidence is genuinely balanced.

## Output Format

**DECISION**: BUY / SELL / HOLD

**Confidence**: High / Medium / Low

**Investment Plan**:
- Action: [specific entry/exit strategy]
- Position Size: [suggested allocation as % of portfolio]
- Time Horizon: [short-term/medium-term/long-term]
- Stop Loss: [price level or % below entry]
- Take Profit: [price level or % above entry]
- Key Risks: [top 3 risks to monitor]

**Reasoning** (2-3 sentences synthesizing why this is the right call):
"""
