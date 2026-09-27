"""Two-phase flash surrogate -- CHEN60492 + the Deep Learning module.

``flash``  the taught flash calculation (Wilson K-values, Rachford-Rice)
``data``   sampling domain, dataset construction, split by mixture, scaling
``nn``     the taught feed-forward network, training loop and physics loss

See ``docs/SOURCE_MAP.md`` for the page or cell behind every equation.
"""
