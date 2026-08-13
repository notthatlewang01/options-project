"""SPX option-chain collection and risk-neutral density estimation.

Components, in strict dependency order:

    ingest    -> store -> curate -> analytics -> cli

Nothing flows backwards. `ingest` is stdlib-only by design and is enforced as
such by tests/test_ingest_is_stdlib_only.py.
"""

__version__ = "0.1.0"
