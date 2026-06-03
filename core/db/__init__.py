"""Oracle connection and query helpers (lazy oracledb import)."""
from .connection import connect, cursor, execute, executemany, fetch_all, replace_table

__all__ = [
    "connect",
    "cursor",
    "fetch_all",
    "execute",
    "executemany",
    "replace_table",
]
