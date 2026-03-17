# algo-trading-bot

Strategy Description

This trading agent uses a mean reversion strategy based on Bollinger Bands.

Indicators used:
1. 20 period moving average
2. 20 period standard deviation
3. Bollinger bands
4. Volume ratio

Trading Logic:

If price falls below the lower Bollinger Band and volume spikes,
the bot buys expecting price to revert to the mean.

If price rises above the upper Bollinger Band,
the bot sells to capture profit.

Risk Management:

The bot uses approximately 20% of available cash per trade.

The agent trades once every 10 seconds
according to the API limits.

The bot includes error handling so it continues running
even if API requests fail.
