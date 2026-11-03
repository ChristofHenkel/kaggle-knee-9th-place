# Source attribution

The SKPv2-derived files are distributed under the accompanying MIT license
(Copyright 2026 Ian Pan). `source_manifest.json` records the imported files and
their original hashes. Project-specific portable integration is by Ian Pan.

`src/skp/callbacks/ema.py` is adapted from timm's `utils/model_ema.py`
(Copyright 2020 Ross Wightman), licensed under Apache-2.0; the original source
URL remains in the file. See `LICENSE.timm`. Other source attribution comments,
including the EMA callback's gale reference, have been retained.

The model relies on timm at runtime rather than bundling its encoder source.
Competition data and trained weights are not distributed under these code licenses.
