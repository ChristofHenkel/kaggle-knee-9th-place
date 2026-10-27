from copy import deepcopy
import hashlib
import json

from labeling.prompts import prompt_016 as report_prompt
from labeling.prompts import prompt_021 as review_prompt
from labeling.prompts import prompt_022 as references


ROOT = references.ROOT
TARGETS = references.TARGETS
cases = references.cases
available_references = references.available_references
EXTRACTION_SCHEMA = deepcopy(references.EXTRACTION_SCHEMA)
for target in TARGETS:
    EXTRACTION_SCHEMA["properties"]["findings"]["properties"][target]["properties"]["panels"]["items"]["enum"] = [
        "01_OVERVIEW.jpg", "02_NAVIGATION.jpg", "03_NAVIGATION.jpg", "04_NAVIGATION.jpg",
    ]

SESSION_RULES = """ONE SESSION, TWO RECORDED STAGES
You receive the full clinical report, but NO case images initially and NO external
report predictions. The initially attached images are ONLY host teaching examples.

STAGE 1: Apply the prompt-016 report instructions below to the complete report.
Before inspecting any case image, call record_report_assessment with language and
all twelve structured findings. This commits your initial probabilities; they
cannot be revised in this session. Do not emit the final response before this call.
Use prompt-016 semantics for this stage; the later image-review instructions govern
stage 2. Do not infer case findings from the host teaching examples or metadata.

STAGE 2: The tool releases the study overview and exact tile/orientation metadata.
Continue IN THIS SAME SESSION. Your own committed assessment is the report-only
starting estimate referenced below. You have the full report and may revisit its
meaning while navigating the MRI. Use render_tiles at least once, up to three
navigation montages, for a total maximum of four case images.
Return final JSON with case_id and findings. The initial assessment is already
saved by the tool; do not duplicate or change it in your final response.
In panels use EXACT filenames only: 01_OVERVIEW.jpg, 02_NAVIGATION.jpg,
03_NAVIGATION.jpg, 04_NAVIGATION.jpg. Put tile IDs in citations and explanations
in visual_evidence; never append descriptions to panel filenames.
"""


def build_requests(row, examples, cfg):
    report = report_prompt.build_prompt(row.Report, examples, cfg)
    case_id = next(key for key, uid in cases(cfg.input_csv).items() if uid == row.StudyInstanceUID)
    directory = ROOT / cfg.image_dir / row.StudyInstanceUID
    catalog = json.loads((directory / "catalog.json").read_text())
    context_path = directory / "audit_context.json"
    images = [ROOT / path for path in cfg.teaching_images]
    review = review_prompt.SYSTEM_PROMPT.replace(
        "{host_guide}", review_prompt.host_guide(cfg.host_guide)
    ).replace(
        "The report-only prompt-016 estimate is supplied as a prior.",
        "Your own committed prompt-016 assessment is the starting estimate."
    ).replace(
        "- The first attached image is the study overview. Subsequent initial attachments\n  are host teaching references. Only tool-returned images belong to this study.",
        "- All initial attachments are host teaching references. Case images are released only by the case tools after the report assessment is committed."
    )
    schema = deepcopy(EXTRACTION_SCHEMA)
    schema["properties"]["case_id"]["enum"] = [case_id]
    return [{
        "system_prompt": SESSION_RULES + "\n\nSTAGE 1 REPORT INSTRUCTIONS:\n"
        + report["system_prompt"] + "\n\nSTAGE 2 IMAGE REVIEW INSTRUCTIONS:\n"
        + review + references.REVIEW_RULES,
        "prompt": f"ANONYMOUS CASE ID: {case_id}\n\n" + report["prompt"]
        + "\n\nACQUIRED SERIES CATALOG (metadata only):\n" + json.dumps(catalog)
        + "\n\nFirst commit your report-only assessment; then inspect and navigate the case.",
        "schema": schema,
        "image_paths": [str(path) for path in images],
        "image_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in images],
        "audit_context_path": str(context_path),
        "audit_context_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
        "host_guide_sha256": hashlib.sha256((ROOT / cfg.host_guide).read_bytes()).hexdigest(),
        "example_ids": [],
    }]


def parse_response(response, cfg):
    result = json.loads(response)
    if set(result) != {"case_id", "findings"} or set(result["findings"]) != set(TARGETS):
        raise ValueError("response keys do not match prompt_023")
    case_id = result["case_id"]
    if not isinstance(case_id, str) or case_id not in cases(cfg.input_csv):
        raise ValueError("invalid anonymous case_id")
    uid = cases(cfg.input_csv)[case_id]
    references = available_references(uid, cfg)
    assessment = json.loads((ROOT / cfg.image_dir / uid / "report_assessment.json").read_text())
    prior = report_prompt.parse_response(json.dumps(assessment), cfg)
    prediction = {"case_id": case_id, "initial_language": assessment["language"]}
    for target in TARGETS:
        for key, value in assessment["findings"][target].items():
            prediction[f"initial_{key}_{target}"] = value
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
