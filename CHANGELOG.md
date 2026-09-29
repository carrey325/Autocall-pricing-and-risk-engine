# Changelog

## 0.3.0rc1 — 2026-09-29

- Integrate the extended Greeks branch into the pricing and risk API:
  Lambda/Omega, Vanna, Vomma/Volga, Charm, and Veta.
- Preserve accrual and event state during time-risk rolls; freeze the PDE
  spatial domain across market bumps. Document numerical limitations.
- Adopt a `src` package layout and one source for distribution/runtime versions.
- Bundle six JSON presets, expose `load_example`, `list_examples`, and
  `config_from_dict`, and add `python -m structured_products`.
- Keep existing product builders, pricing functions, configuration classes,
  and result objects. Add typing metadata and public API tutorials.
- Verify wheel and source installations in fresh environments outside the
  checkout; add Linux, Windows, and macOS distribution checks to CI.

This candidate is distributed through GitHub only. No PyPI publication or
GitHub Release is created by the project workflows.

## 0.2.0

- Modular Snowball/FCN products, single-asset MC/PDE, and multi-asset worst-of MC.
- Pricing diagnostics, basic Greeks, fair coupon, strict JSON configuration,
  CLI, and numerical validation.
