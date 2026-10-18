"""Reproduce the full installed research entrypoint; no notebook or network."""
from structured_products.cli import main

if __name__=='__main__':
    import sys
    raise SystemExit(main(['research',*sys.argv[1:]]))
