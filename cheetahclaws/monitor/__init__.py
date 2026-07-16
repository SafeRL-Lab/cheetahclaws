"""
monitor/ — AI Monitor + Decision Assistant for CheetahClaws.

Provides long-running subscription monitoring with AI summarization.

Topics:
  ai_research     — arxiv cs.AI/cs.LG latest papers
  stock_TICKER    — stock price, change, news (e.g. stock_TSLA)
  crypto_SYMBOL   — crypto price + market data (e.g. crypto_BTC)
  world_news      — top world news via RSS
  custom:QUERY    — monitor any custom search query

Usage (slash commands):
  /subscribe ai_research
  /subscribe stock_TSLA
  /subscribe crypto_BTC
  /subscriptions
  /unsubscribe ai_research
  /monitor run [topic]
  /monitor start
  /monitor stop
  /monitor status
"""
