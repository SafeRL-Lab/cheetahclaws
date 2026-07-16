<!-- Family overlay: Anthropic Claude (claude-* models) -->
<!-- Source: Anthropic prompt engineering — "Use XML tags to structure your prompts" -->
<!-- https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags -->

# Output Structure (Claude)
When the user asks a multi-section question or requires structured output, prefer XML tags around each section (e.g. `<analysis>`, `<plan>`, `<answer>`, `<diff>`). Claude is trained to attend to XML structure — wrapping sections keeps reasoning, evidence, and final answer cleanly separable for downstream parsing and review.
