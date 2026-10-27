# OAI to RSNA: seven-class pretraining label mapping

This documents the labels used by **v700, v701 and v702** in the completed
[OAI dataset](./input/oai/). It describes the implemented mapping, not an
alternative proposal. The experiments share the same targets and masks;
v701 changes sampling and v702 changes the backbone.

## 1. What the target means

Each usable target is a soft estimate of:

> The probability that an annotator would mark the class positive from the
> knee MRI under the RSNA competition rubric.

It is **not** a generic probability of disease, an OAI severity score divided
by its maximum, or an annotation-confidence weight.

The reference is the
[RSNA Knee Abnormality Detection annotation rubric](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343).
It favors specificity: ambiguous/borderline findings are negative. Some OAI
abnormalities therefore map to low probabilities even though they are not normal.

**The numerical probabilities below are manually chosen heuristics.** They have
not been calibrated from paired OAI and RSNA-rubric readings, and they do not
guarantee an improvement in ROC AUC. Their purpose is to preserve useful
rubric-aligned distinctions without pretending that the two annotation systems
are identical.

## 2. Source data, joins and output format

### Label sources

The current dataset uses the **main knee MRI MOAKS tables**:
`kMRI_SQ_MOAKS_BICLxx.txt`, where `xx` identifies the visit.

| Prefix | Visit |
|---|---|
| `V00` | Baseline |
| `V01` | 12 months |
| `V03` | 24 months |
| `V05` | 36 months |
| `V06` | 48 months |
| `V08` | 72 months |
| `V10` | 96 months |

In the field names below, replace `Vxx` with the appropriate prefix.

- Match imaging and labels by **participant ID + knee side + visit**.
- OAI side codes are `1 = right`, `2 = left`.
- `READPRJ` identifies the reading project. Multiple projects can assess the
  same knee/visit; their rows are reconciled rather than treated as extra studies.
- Use DICOM headers to obtain the final `StudyInstanceUID`.
- Left/right image canonicalization does **not** swap medial/lateral labels.

The current cohort does **not** add POMA-only or WORMS-only knees. Non-knee
assessments, radiographic KL/JSN scores, quantitative cartilage measurements,
clinical outcomes and teacher predictions are not used to construct these targets.

### Final label schema

[labels.csv](./input/oai/labels.csv) has one row per knee examination:

```text
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
```

- A value in `[0,1]` is a training target.
- **`-1` means ignore this study-target cell.**
- A masked cell is unknown/excluded, not a negative example.
- All twelve output columns remain present, but only the seven classes below
  receive OAI supervision.
- There are **no confidence columns or confidence multipliers**.

## 3. Medial Meniscus and Lateral Meniscus

### Rubric interpretation

A positive requires definite surface-reaching tear signal on at least two
images, or a morphological abnormality such as a truncated/diminutive meniscus
or displaced fragment. Intrasubstance degeneration without surface contact
does not qualify.

OAI tear categories provide indirect evidence of this criterion; the label
mapping itself cannot recheck the two-image requirement.

### Fields

| Target | Anterior horn | Body | Posterior horn | Posterior root tear |
|---|---|---|---|---|
| Medial Meniscus | `VxxMMTMA` | `VxxMMTMB` | `VxxMMTMP` | `VxxMMRTM` |
| Lateral Meniscus | `VxxMMTLA` | `VxxMMTLB` | `VxxMMTLP` | `VxxMMRTL` |

### Regional morphology mapping

| Raw OAI code and meaning | Regional probability |
|---|---:|
| `0: Normal` | 0.02 |
| `1: Signal abnormality` | 0.05 |
| `2: Radial tear` | 0.92 |
| `3: Horizontal tear` | 0.92 |
| `4: Vertical tear` | 0.92 |
| `5: Complex tear` | 0.97 |
| `6: Partial maceration` | 0.97 |
| `7: Progressive partial maceration` | 0.97 |
| `8: Complete maceration` | 0.97 |

Posterior root tear `1: Yes` supplies additional positive evidence of **0.97**.
Root tear `0: No` does not supply a positive; it is needed to establish a
fully covered negative.

Codes 2, 3 and 4 are different tear types, **not an increasing severity scale**.
Signal abnormality, extrusion, hypertrophy or meniscal cysts alone do not
manufacture a tear-positive target.

### Combine regions within one reading

1. If any horn has a recognized tear/maceration code `2..8`, or the root is
   explicitly positive, use the **maximum positive probability**.
2. Otherwise, if all three horns are scored `0` or `1` **and** the root is
   explicitly `0`, use the maximum horn probability: `0.02` or `0.05`.
3. Otherwise, mask the target (`-1` on export).

Thus, a known positive can remain usable despite another missing region.
A negative needs adequate coverage.

Examples:

| Horn codes `(anterior, body, posterior)` | Root | Output |
|---|---|---:|
| `(0, 0, 0)` | `0` | 0.02 |
| `(1, 0, 1)` | `0` | 0.05 |
| `(0, 3, 1)` | `0` | 0.92 |
| `(missing, missing, 6)` | missing | 0.97 |
| `(missing, missing, missing)` | `1` | 0.97 |
| `(0, 0, missing)` | `0` | -1 |
| `(0, 0, 0)` | missing | -1 |

The medial and lateral targets are computed independently.

## 4. Medial OA, Lateral OA and PF OA

### Rubric interpretation

These classes concern **a moderate/large area of high-grade cartilage loss**:
roughly 1 cm or greater in extent and greater than 50% loss of cartilage
thickness, with or without associated subchondral marrow changes.

A general OA diagnosis, osteophytes, joint-space narrowing or a marrow lesion
alone does not establish this MRI criterion.

### MOAKS cartilage codes are two-part categories

The `VxxMCM...` fields combine two scores:

- **Before the decimal:** proportion of the subregion's surface affected by
  any cartilage loss, whether partial or full thickness.
- **After the decimal:** proportion of the subregion's surface affected by
  full-thickness loss.

Area categories are `0 = none`, `1 = small (approximately <10%)`,
`2 = 10-75%`, and `3 = >75%`. The text export commonly labels category 1 as
`1-10%`. An integer such as `2` means `2.0`, not `2.2`.

For example:

```text
2: 10-75% area, no full thickness
2.2: 10-75% area, 10-75% full thickness
```

**No full-thickness loss does not mean no high-grade loss.** A lesion can lose
more than 50% of its thickness without losing all of it. MOAKS does not resolve
that depth distinction here. Also, percentage of a subregion is not an exact
measurement in centimetres. This is why the intermediate mappings remain uncertain.

### Regional cartilage mapping

| OAI code | Area with any loss | Area with full-thickness loss | Regional probability |
|---|---|---|---:|
| `0` | None | None | 0.02 |
| `1` / `1.0` | Small, approximately <10% | None | 0.10 |
| `1.1` | Small, approximately <10% | Small, approximately <10% | 0.35 |
| `2` / `2.0` | 10-75% | None | 0.25 |
| `2.1` | 10-75% | Small, approximately <10% | 0.60 |
| `2.2` | 10-75% | 10-75% | 0.90 |
| `3` / `3.0` | >75% | None | 0.40 |
| `3.1` | >75% | Small, approximately <10% | 0.70 |
| `3.2` | >75% | 10-75% | 0.95 |
| `3.3` | >75% | >75% | 0.98 |

Do not sort these as an ordinary severity number: `3.0` maps below `2.2`
because the former does not establish the depth of loss.

### Compartment fields

| Target | Exact fields |
|---|---|
| **Medial OA** | `VxxMCMFMC`, `VxxMCMFMP`, `VxxMCMTMA`, `VxxMCMTMC`, `VxxMCMTMP` |
| **Lateral OA** | `VxxMCMFLC`, `VxxMCMFLP`, `VxxMCMTLA`, `VxxMCMTLC`, `VxxMCMTLP` |
| **PF OA** | `VxxMCMFMA`, `VxxMCMFLA`, `VxxMCMPM`, `VxxMCMPL` |

Anatomical interpretation:

- Medial: central/posterior medial femur and anterior/central/posterior medial tibia.
- Lateral: central/posterior lateral femur and anterior/central/posterior lateral tibia.
- PF: medial/lateral anterior femur (trochlea) and medial/lateral patella.

**`FMA` and `FLA` belong to PF OA**, despite containing "medial"/"lateral" in
their anatomical names. They are not assigned to the tibiofemoral OA targets.

### Combine regions within one reading

1. If any region is `2.2`, `3.2` or `3.3`, use the maximum probability among
   those strong-positive regions, even if another region is missing.
2. Otherwise, require **every region in that compartment** to have a recognized
   cartilage code, then take the maximum regional probability.
3. If neither condition is satisfied, mask the compartment target.

The incomplete-coverage exception applies **only to `2.2`, `3.2`, `3.3`**.
For example, a known `3.1` region with other missing regions is still masked;
the current implementation does not accept every probability above 0.5 as
sufficient evidence under partial coverage.

Examples:

| Compartment's region codes | Output |
|---|---:|
| All `0` | 0.02 |
| One `2`, all others `0` | 0.25 |
| One `3.1`, all others valid and no stronger region | 0.70 |
| One `2.2`, others missing | 0.90 |
| One `3.3`, others `0` | 0.98 |
| One `3.1`, another region missing | -1 |
| Only one region `0`, others missing | -1 |

### Follow-up codes `0.5` and `-0.5`

These mean **within-grade worsening** and **within-grade improvement**,
respectively. They are not absolute cartilage grades or target probabilities.

The builder:

1. Looks up the preceding stored regional grade for the same
   **participant + side + READPRJ + field**, in visit order.
2. If that grade is in the recognized cartilage mapping, reuses the grade.
   The direction of within-grade change does not change the soft target.
3. If no usable predecessor exists, treats the region as missing.
4. Records each resolution in the label provenance.

This is not general forward filling. A missing/invalid preceding record can
break the chain; grades are not borrowed from another project, knee or future
visit. Compartment-level missing-region rules still apply afterward.

## 5. Effusion

**Source field:** `VxxMEFFWK`, whole-knee effusion.

The competition requires **moderate or large fluid distension**. Trace/mild
fluid is not automatically positive.

| Raw OAI value | Probability |
|---|---:|
| `0: None` | 0.02 |
| `1: Mild` | 0.08 |
| `2: Moderate` | 0.85 |
| `3: Severe` | 0.98 |
| Missing or unrecognized value | -1 |

OAI uses non-contrast imaging. Its documentation notes that the effusion score
can include synovial tissue as well as fluid, so even this relatively close
mapping is not an identical annotation protocol.

## 6. Synovitis

**Source field:** `VxxMSYIC`, intercondylar/Hoffa synovitis.

The competition describes inflammation and thickening of the synovial lining.
OAI's score is based on Hoffa/intercondylar signal abnormality, which is an
**imperfect proxy** for that criterion.

| Raw OAI value | Probability |
|---|---:|
| `0: None` | 0.10 |
| `1: Mild` | 0.25 |
| `2: Moderate` | 0.70 |
| `3: Severe` | 0.90 |
| Missing or unrecognized value | -1 |

If the reading's comment matches an explicit alternative explanation such as
`not pure synovitis`, `lipodystrophy`, or `Hoffa ... syndrome`, that reading's
synovitis contribution is masked. Another compatible usable reading project can
still provide the target under the project-combination rules below.

An effusion value is **not** copied into the synovitis label, or vice versa.
Synovitis has no special loss downweighting in the current experiments.

## 7. Missing values and multiple reading projects

### Parse raw OAI values

For exported strings such as `3: Horizontal tear`, parse the number before
the colon. Blank values, text that is not a numeric code, and SAS
special-missing codes beginning with `.` are missing.

```text
0: Normal                             -> recognized code 0
3: Horizontal tear                    -> recognized code 3
.: Missing Form/Incomplete Workbook   -> missing
.R: ...                               -> missing
```

Parsing a number does not make it valid for every field. For example, an
effusion code outside `{0,1,2,3}` is not converted using a generic `code > 0`
test. Unrecognized codes behave as missing when determining the target.

### Reconcile `READPRJ` rows per study-target

Apply the class/region mapping separately to each reading project first.
For a particular target:

1. Ignore project contributions already masked by that mapping.
2. If no usable contribution remains, output `-1`.
3. Let `low` and `high` be the minimum and maximum usable probabilities.
4. Mask the target if either:
   - `low < 0.5 <= high`; or
   - `high - low > 0.200001` (the small tolerance avoids floating-point noise).
5. Otherwise, output the **arithmetic mean** of the usable probabilities.

Examples:

| Project probabilities | Final target |
|---|---:|
| `0.92`, `0.97` | 0.945 |
| `0.02`, `0.05` | 0.035 |
| `0.02`, `0.92` | -1 |
| `0.10`, `0.40` | -1 |
| Missing, `0.85` | 0.85 |
| All missing | -1 |

This creates **one target per study**, not one training example per reading
project. Final values are serialized to 12 significant digits.

### Conservative study exclusions

The full cohort permits any knee with at least one usable active target.
Individual absent/conflicting targets remain `-1`; the other targets can train.

Separately, the current builder excludes the entire knee/visit if any reading's
general or technical comment matches its technical-quality filter: unscorable
or unevaluable findings, motion, artifacts, fat-suppression failure, missing
images, poor quality, or hardware. This is a conservative text filter; notes can
refer to another visit, so it is not a perfect visit-specific quality assessment.
Its exact patterns are recorded in the builder.

The imaging manifest additionally requires the five selected same-date knee
sequences with completed `Y`/`YD` inventory QC. After conversion, studies with
an invalid selected archive or spatial stack are removed from the final labels.
Conversion exclusions do not change surviving target probabilities.

## 8. Globally masked output classes

These columns are always `-1` in the current OAI training labels, even if a
source table has some related observations:

| Output | Why not supervised in this dataset |
|---|---|
| ACL | Too few clearly rubric-compatible positives under the chosen conservative rarity policy; a generic partial tear does not establish >50% fibre disruption. |
| MCL | No adequate high-grade acute-tear positive set in the inspected sources; low-grade signal changes are not sufficient. |
| Baker's | Main MOAKS records presence without size, whereas the rubric requires moderate/large size; size-resolved positives in the inspected WORMS subset were sparse. |
| Contusion | Degenerative marrow lesions do not establish impact-related bone contusions. |
| Fracture | No systematic compatible acute-fracture labels with reliable positive/negative coverage. |

The earlier rarity decision used fewer than 100 distinct clearly positive
participants as a conservative threshold. This is a dataset-design choice,
not a learned threshold or a rule recomputed per minibatch.

## 9. How these labels enter training

All three experiments use **unweighted masked BCE**:

```python
valid = targets >= 0
if not valid.any():
    raise ValueError("No usable target cells in this batch")

main = binary_cross_entropy_with_logits(logits[valid], targets[valid])
aux = binary_cross_entropy_with_logits(auxiliary_logits[valid], targets[valid])
loss = main + 0.2 * aux
```

Both means are over valid study-target cells only. The implementation also
preserves equal per-cell contribution across gradient-accumulation microbatches.
Do not pass `-1` into BCE or convert it into a negative example.

- No confidence weights or per-class loss multipliers are used.
- The factor `0.2` scales the entire auxiliary objective; it is not confidence.
- Probabilities stay soft during training. A `0.5` cutoff used in count summaries
  does not turn the training targets into hard labels.
- Gold-58 validation reads the competition's hard labels, not these mapped OAI
  targets. It is informational and does not select checkpoints.

## 10. Actual output example and current coverage

One real row in the completed dataset:

```csv
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
1.3.12.2.1107.5.2.13.20579.4.0.3093934819662212,-1,-1,0.05,0.97,0.02,-1,-1,0.85,0.25,-1,-1,-1
```

The final build contains **13,719 studies**. These counts describe usable
annotations, not positive prevalence:

| Active class | Usable cells | Masked cells |
|---|---:|---:|
| Medial Meniscus | 13,437 | 282 |
| Lateral Meniscus | 13,556 | 163 |
| Medial OA | 13,300 | 419 |
| Lateral OA | 13,283 | 436 |
| PF OA | 13,149 | 570 |
| Effusion | 11,036 | 2,683 |
| Synovitis | 11,023 | 2,696 |

There are **88,784 usable target cells** across all twelve output columns;
the five globally disabled columns contribute none.

## 11. Provenance and implementation references

- [Frozen mapping policy used for this dataset](./input/oai/provenance/mapping_policy.json).
- [Final label CSV](./input/oai/labels.csv).
- [Final study-to-participant/side/visit manifest](./input/oai/study_manifest.csv).
- [Completed dataset summary](./input/oai/dataset.json).
- [Label mapping implementation](./tmp/oai_50_knees/build_sample.py):
  `number`, `label_fields`, `map_label`, `combine_labels`, `load_labels`.
- [Full-cohort export implementation](./tmp/oai_full_manifest/build_manifest.py).
- [Per-study raw field evidence, project IDs and mapping reasons](./tmp/oai_full_manifest/label_evidence.jsonl).
  This is the pre-conversion manifest: join to final `labels.csv` to restrict it
  to studies that passed conversion.
- [Pre-conversion label/QC exclusions](./tmp/oai_full_manifest/excluded_studies.csv).
- [Conversion exclusions](./input/oai/excluded_studies.csv).
- [Shared training loss](./scripts/train_v700_2p5d_oai_convnext_remote01.py):
  `loss_terms` and `loss_fn`.

The OAI source contains structured scoring fields plus occasional short reader
comments, **not full diagnostic MRI reports**. Comments such as `none` mean
"no additional comment"; they are not evidence that all seven targets are negative.
