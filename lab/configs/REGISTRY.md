# Config registry

## Approaches

### Approach 1 — Slow-R50 study classifier

Next variant: 97.

| Config | Parent | Definition |
|---|---|---|
| cfg_1_85 | — | L85, eight slots, official series fields only, no laterality normalization |
| cfg_1_91 | cfg_1_85 | Twelve slots and L85/OOF85 fusion |

### Approach 8 — ConvNeXt study classifier

| Config | Parent | Definition |
|---|---|---|
| cfg_8_29 | cfg_1_85 | Tiny, native triplets, L85 and OOF85 loss weights |
| cfg_8_31 | cfg_8_29 | Tiny, shared cache, L85/OOF829 fusion |
| cfg_8_32 | cfg_8_29 | Small, native triplets, L85 and OOF85 loss weights |

## Legacy

OA preparation/training/transfer: cfg_18_7, cfg_18_transfer.
