"""
Post-trade reflection mechanism.

After a trading decision's outcome is known (profit or loss),
this module generates reflections for each agent component and
stores lessons learned in BM25 memory for future reference.

Inspired by TradingAgents' Reflector pattern.
"""
from __future__ import annotations

from .memory import get_memory


def get_reflection_prompt(
    component: str,
    symbol: str,
    trade_date: str,
    decision: str,
    outcome: str,
    returns: float,
    context: str = "",
) -> str:
    """Generate a reflection prompt for a specific agent component.

    Args:
        component: agent role (bull_researcher, bear_researcher, trader,
                   risk_judge, portfolio_manager)
        symbol: instrument traded
        trade_date: when the decision was made
        decision: the original recommendation/analysis
        outcome: what actually happened
        returns: profit/loss amount
        context: additional market context

    Returns:
        Prompt for the LLM to generate a reflection.
    """
    correct = "CORRECT" if returns > 0 else "INCORRECT"
    result_desc = f"profit of ${returns:,.2f}" if returns > 0 else f"loss of ${abs(returns):,.2f}"

    component_labels = {
        "bull_researcher": "Bull Research Analyst",
        "bear_researcher": "Bear Research Analyst",
        "trader": "Trader",
        "risk_judge": "Research Judge",
        "portfolio_manager": "Portfolio Manager",
    }
    role = component_labels.get(component, component)

    return f"""You are reviewing a past trading decision to extract lessons for improvement.

**Role**: {role}
**Symbol**: {symbol}
**Decision Date**: {trade_date}
**Outcome**: The decision was {correct}, resulting in a {result_desc}

## Original Decision/Analysis
{decision[:1000]}

## What Actually Happened
{outcome}

## Additional Context
{context}

## Reflection Task

Analyze this {correct.lower()} decision and extract actionable lessons:

1. **What went {'right' if returns > 0 else 'wrong'}?**
   - Which factors in your analysis were {'validated' if returns > 0 else 'invalidated'}?
   - Which indicators or data points were most {'reliable' if returns > 0 else 'misleading'}?

2. **Contributing Factors**:
   - Market conditions at the time
   - Technical indicator readings
   - Fundamental data quality
   - News/sentiment accuracy

3. **{'What to Repeat' if returns > 0 else 'Corrective Action'}**:
   - {'Which analytical approaches should be reinforced?' if returns > 0 else 'What specific changes would improve future decisions?'}
   - {'What patterns correctly identified the opportunity?' if returns > 0 else 'What warning signs were missed?'}

4. **Condensed Lesson** (IMPORTANT — this will be stored in memory):
   Write a single paragraph (~100 words) summarizing:
   - The market situation
   - The decision made
   - The outcome
   - The key takeaway for future similar situations

Start the condensed lesson with "LESSON:" on its own line.
"""


def extract_lesson(reflection_response: str) -> tuple[str, str]:
    """Extract the condensed lesson from a reflection response.

    Returns:
        (situation_summary, lesson_text) tuple for memory storage.
    """
    # Find the LESSON: section
    lines = reflection_response.split("\n")
    lesson_start = -1
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("LESSON:"):
            lesson_start = i
            break

    if lesson_start >= 0:
        # Collect all text after LESSON: until end or next section
        lesson_lines = []
        first_line = lines[lesson_start].split(":", 1)
        if len(first_line) > 1 and first_line[1].strip():
            lesson_lines.append(first_line[1].strip())
        for line in lines[lesson_start + 1:]:
            if line.strip().startswith("#") or line.strip().startswith("---"):
                break
            if line.strip():
                lesson_lines.append(line.strip())
        lesson = " ".join(lesson_lines)
    else:
        # Fallback: use last paragraph
        paragraphs = [p.strip() for p in reflection_response.split("\n\n") if p.strip()]
        lesson = paragraphs[-1] if paragraphs else reflection_response[:200]

    # Split into situation (first sentence) and recommendation (rest)
    sentences = lesson.split(". ")
    if len(sentences) >= 2:
        situation = sentences[0] + "."
        recommendation = ". ".join(sentences[1:])
    else:
        situation = lesson[:len(lesson)//2]
        recommendation = lesson[len(lesson)//2:]

    return situation, recommendation


def store_reflection(
    component: str,
    situation: str,
    recommendation: str,
    outcome: str,
    date: str = "",
    symbol: str = "",
) -> None:
    """Store a reflection lesson in the component's BM25 memory."""
    memory = get_memory(component)
    memory.add(
        situation=situation,
        recommendation=recommendation,
        outcome=outcome,
        date=date,
        symbol=symbol,
    )


def reflect_all_components(
    symbol: str,
    trade_date: str,
    decisions: dict[str, str],
    outcome: str,
    returns: float,
) -> list[dict]:
    """Generate reflection prompts for all components.

    Args:
        decisions: {"bull_researcher": text, "bear_researcher": text, ...}
        outcome: what happened
        returns: profit/loss

    Returns:
        List of {"component", "prompt"} dicts for LLM processing.
    """
    prompts = []
    for component, decision_text in decisions.items():
        prompt = get_reflection_prompt(
            component=component,
            symbol=symbol,
            trade_date=trade_date,
            decision=decision_text,
            outcome=outcome,
            returns=returns,
        )
        prompts.append({"component": component, "prompt": prompt})
    return prompts
