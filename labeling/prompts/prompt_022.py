from copy import deepcopy
import json
from functools import lru_cache

import pandas as pd

from labeling.prompts import prompt_021 as base


ROOT = base.ROOT
TARGETS = base.TARGETS
EXTRACTION_SCHEMA = deepcopy(base.EXTRACTION_SCHEMA)
EXTRACTION_SCHEMA["properties"]["case_id"] = {"type": "string"}
EXTRACTION_SCHEMA["required"].append("case_id")
for target in TARGETS:
    finding = EXTRACTION_SCHEMA["properties"]["findings"]["properties"][target]
    finding["properties"]["citations"] = {
        "type": "array", "items": {"type": "string"},
    }
    finding["properties"]["inference_scope"] = {
        "type": "string", "enum": [
            "local_positive", "adequately_sampled_negative", "unresolved",
        ],
    }
    finding["required"].extend(["citations", "inference_scope"])

REVIEW_RULES = """
EXACT CASE REFERENCES AND THRESHOLD REASONING
Return the supplied anonymous case_id exactly. The study overview AND navigation
images belong to this case. The two host teaching plates are rubric references,
not this case, and cannot be cited as case evidence.

Each native tile has a unique citation_id. P01 is the study overview; P02, P03,
P04 are successive navigation images. T01, T02, ... are the tile positions in
reading order. Metadata supplies one-based row and column: the overview has six
columns and each navigation montage has four. Use only IDs actually supplied by overview metadata or successful tool
results. Return exact IDs in citations for every concrete image claim. Never
invent intermediate indices or cite an inclusive range containing unseen slices.
Use citation IDs in visual_evidence rather than free-form slice ranges.

Use supplied frozen laterality and patient orientation to identify compartments.
Frozen laterality is an anatomical prediction, not a finding label. Confirm it
with fibular and collateral-ligament anatomy. A patient R marker does not always
mean the lateral side: patient right is lateral in a right knee, medial in a left
knee. The catalog states the sagittal index progression. If anatomy and predicted
laterality conflict, keep compartment identity unresolved instead of guessing.

For MCL, some continuous fibers do NOT exclude disruption of more than 50 percent
of fibers. Assess the host's high-grade partial threshold, not only whether a
complete gap is present. Periligamentous edema alone does not establish grade.
If the relevant fibers cannot be sufficiently resolved, retain the report prior.

For Baker's, separately establish characteristic gastrocnemius-semimembranosus
location/neck and a moderate or large residual collection. Posterior tracking
fluid alone does not establish a qualifying cyst. Rupture or extensive leaked
fluid does not establish residual collection size. If identity or threshold
eligibility is unresolved, do not promote it to a confident qualifying positive.

For Effusion, distinguish bright cartilage and ordinary recess fluid from fluid
that actually distends the joint moderately or substantially. Mere fluid presence
is insufficient. Inspect joint recess morphology against the host definitions.

inference_scope describes the evidence for the target-level probability update:
- local_positive: directly identifiable positive morphology satisfies the target
  criterion at a cited location. Local positive evidence can be decisive even
  when the rest of the study is incompletely covered.
- adequately_sampled_negative: the relevant structure and qualifying lesion extent
  are sufficiently assessed with appropriate planes, contrast, and adjacent native
  images to support a negative inference. Set coverage=adequate. A visible normal
  patch cannot establish absence throughout an unreviewed structure or compartment.
- unresolved: visibility, anatomy, lesion identity, or criterion eligibility remains
  unresolved. Retain the report-only probability exactly.

A probability decrease requires adequately_sampled_negative with coverage=adequate
and actual criterion-specific negative evidence. Limited coverage plus failure to
see a lesion cannot lower either an omitted finding prior or a positive report
estimate. A probability increase requires local_positive and actual qualifying
positive evidence. All adjustments require exact citations. If image_relation is
inconclusive OR inference_scope is unresolved, retain the report prior exactly.
This is an evidence rule, not a fixed probability cap or cohort-derived prior.
"""


@lru_cache
def cases(path):
    frame = pd.read_csv(ROOT / path, dtype={"StudyInstanceUID": str})
    return {
        f"case_{index + 1:03d}": uid
        for index, uid in enumerate(frame.StudyInstanceUID)
    }


def tile_reference(panel_number, ordinal):
    columns = 6 if panel_number == 1 else 4
    return {
        "citation_id": f"P{panel_number:02d}T{ordinal:02d}",
        "row": (ordinal - 1) // columns + 1,
        "column": (ordinal - 1) % columns + 1,
    }


def build_requests(row, examples, cfg):
    request = deepcopy(base.build_requests(row, examples, cfg)[0])
    case_id = next(key for key, uid in cases(cfg.input_csv).items() if uid == row.StudyInstanceUID)
    directory = ROOT / cfg.image_dir / row.StudyInstanceUID
    manifest = json.loads((directory / "manifest.json").read_text())
    catalog = json.loads((directory / "catalog.json").read_text())
    sides = manifest["frozen_laterality_predictions"]
    references = [{
        **tile_reference(1, index + 1),
        "series_key": tile["series_key"], "native_index": tile["index"],
    } for index, tile in enumerate(manifest["panels"][0]["tiles"])]
    progression = []
    for series in catalog["series"]:
        if series["plane"] != "Sagittal":
            continue
        direction = series["slice_index_direction"][:1]
        if len(sides) == 1 and sides[0] in ("R", "L") and direction in ("R", "L"):
            value = "medial toward lateral" if sides[0] == direction else "lateral toward medial"
        else:
            value = "unresolved; identify compartment from visible anatomy"
        progression.append({
            "series_key": series["series_key"],
            "increasing_native_index": value,
            "patient_direction": series["slice_index_direction"],
        })
    request["system_prompt"] = request["system_prompt"].replace(
        "Only tool-returned images belong to this study.",
        "The overview and tool-returned images belong to this study.",
    ) + REVIEW_RULES
    request["prompt"] = (
        f"ANONYMOUS CASE ID: {case_id}\n\n" + request["prompt"]
        + "\n\nCASE ORIENTATION AND EXACT OVERVIEW REFERENCES:\n"
        + json.dumps({
            "frozen_laterality_predictions": sides,
            "sagittal_index_progression": progression,
            "overview_references": references,
        }, ensure_ascii=False, allow_nan=False)
    )
    request["schema"] = deepcopy(EXTRACTION_SCHEMA)
    request["schema"]["properties"]["case_id"]["enum"] = [case_id]
    return [request]


def available_references(uid, cfg):
    directory = ROOT / cfg.image_dir / uid
    manifest = json.loads((directory / "manifest.json").read_text())
    panels = manifest["panels"][:1]
    navigation = directory / "navigation_requests.jsonl"
    if navigation.exists():
        panels += [json.loads(line)["panel"] for line in navigation.read_text().splitlines()]
    references = {}
    for panel in panels:
        number = int(panel["file"].split("_", 1)[0])
        for ordinal, _ in enumerate(panel["tiles"], 1):
            references[tile_reference(number, ordinal)["citation_id"]] = panel["file"]
    return references


def parse_response(response, cfg):
    result = json.loads(response)
    if set(result) != {"case_id", "findings"} or set(result["findings"]) != set(TARGETS):
        raise ValueError("response keys do not match prompt_022")
    case_id = result["case_id"]
    if not isinstance(case_id, str) or case_id not in cases(cfg.input_csv):
        raise ValueError("invalid anonymous case_id")
    uid = cases(cfg.input_csv)[case_id]
    references = available_references(uid, cfg)
    prior = base.report_predictions(cfg.report_predictions_csv).loc[uid]
    prediction = {"case_id": case_id}
    for target in TARGETS:
        finding = result["findings"][target]
        fields = EXTRACTION_SCHEMA["properties"]["findings"]["properties"][target]["properties"]
        if set(finding) != set(fields):
            raise ValueError(f"invalid finding fields for {target}")
        for key, value in finding.items():
            field = fields[key]
            if "enum" in field and value not in field["enum"]:
                raise ValueError(f"invalid {key} for {target}")
            if field["type"] == "string" and not isinstance(value, str):
                raise ValueError(f"invalid text {key} for {target}")
            if field["type"] == "array" and (
                not isinstance(value, list) or any(not isinstance(item, str) for item in value)
            ):
                raise ValueError(f"invalid list {key} for {target}")
        probability = finding["probability"]
        if type(probability) not in (int, float) or not 0 <= probability <= 1:
            raise ValueError(f"invalid probability for {target}")
        if any(citation not in references for citation in finding["citations"]):
            raise ValueError(f"uncited or unrendered tile for {target}")
        filenames = {references[citation] for citation in finding["citations"]}
        if any(panel not in set(references.values()) for panel in finding["panels"]):
            raise ValueError(f"unrendered panel for {target}")
        if not filenames <= set(finding["panels"]):
            raise ValueError(f"citation panels missing for {target}")
        delta = probability - float(prior[f"probability_{target}"])
        unresolved = finding["image_relation"] == "inconclusive" or finding["inference_scope"] == "unresolved"
        if unresolved and abs(delta) > 1e-12:
            raise ValueError(f"unresolved evidence must retain the prior for {target}")
        if abs(delta) > 1e-12 and not finding["citations"]:
            raise ValueError(f"probability adjustment requires exact citations for {target}")
        if delta < -1e-12 and (
            finding["inference_scope"] != "adequately_sampled_negative" or finding["coverage"] != "adequate"
        ):
            raise ValueError(f"negative adjustment requires adequate target assessment for {target}")
        if delta > 1e-12 and finding["inference_scope"] != "local_positive":
            raise ValueError(f"positive adjustment requires local positive evidence for {target}")
        for key, value in finding.items():
            prediction[f"{key}_{target}"] = json.dumps(value) if isinstance(value, list) else value
        prediction[f"raw_probability_{target}"] = float(probability)
        prediction[f"score_{target}"] = float(probability)
    return prediction
