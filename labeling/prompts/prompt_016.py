import json
import re
import unicodedata
import hashlib
from functools import lru_cache
from pathlib import Path

from labeling.prompts.prompt_012 import TARGETS


HOST_CRITERIA = {'ACL': 'A high-grade partial or full-thickness tear of the anterior cruciate ligament, meaning complete discontinuity of the ligament, or more than 50 percent of fibers disrupted, with or without secondary signs such as characteristic pivot-shift bone contusions. Mild signal change, degeneration, or thickening without discontinuity is graded negative.', 'MCL': 'A high-grade partial or complete acute tear of the medial collateral ligament, with disrupted fibers and edema within and adjacent to the ligament. Low-grade sprains and chronic or remote stress changes are graded negative.', 'Medial Meniscus': 'Abnormal signal that definitely contacts the meniscal surface on at least two images, or a morphologic abnormality such as a truncated, diminutive, or displaced fragment, involving the medial meniscus. Intrasubstance degeneration that does not reach the surface is negative.', 'Lateral Meniscus': 'The same criteria applied to the lateral meniscus.', 'Medial OA': 'A moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, defined as greater than 50 percent of cartilage thickness, in the medial compartment, with or without underlying subchondral marrow changes.', 'Lateral OA': 'The same criteria applied to the lateral compartment.', 'PF OA': 'The same criteria applied to the patellofemoral compartment.', 'Effusion': 'A moderate or large amount of fluid distending the joint.', 'Synovitis': 'Inflammation and thickening of the synovial lining of the joint.', "Baker's": 'A moderate or large fluid collection in the characteristic location behind the knee.', 'Contusion': 'A bone contusion, seen as bone marrow edema-like signal from impact, without a discrete fracture line.', 'Fracture': 'An acute cortical break or fracture line.'}
ASSERTION = ['present', 'absent', 'equivocal', 'not_mentioned']
SEVERITY = ['low', 'moderate', 'high', 'unqualified', 'not_applicable']
EXTENT = ['low_grade', 'partial', 'high_grade_partial', 'complete', 'unspecified', 'not_applicable']
CHRONICITY = ['acute', 'chronic', 'unspecified', 'not_applicable']
SURFACE = ['reaches_surface', 'intrasubstance_only', 'unspecified', 'not_applicable']
FRACTURE_LINE = ['present', 'absent', 'unspecified', 'not_applicable']
_FIELD = {'probability': {'type': 'number'}, 'assertion': {'type': 'string', 'enum': ASSERTION}, 'severity': {'type': 'string', 'enum': SEVERITY}, 'extent': {'type': 'string', 'enum': EXTENT}, 'chronicity': {'type': 'string', 'enum': CHRONICITY}, 'surface_contact': {'type': 'string', 'enum': SURFACE}, 'discrete_fracture_line': {'type': 'string', 'enum': FRACTURE_LINE}, 'evidence': {'type': 'string'}}
TARGET_FIELDS = {t: ['probability', 'assertion', 'severity', 'extent', 'chronicity', 'surface_contact', 'discrete_fracture_line', 'evidence'] for t in TARGETS}

def _item(fields):
    return {'type': 'object', 'additionalProperties': False, 'properties': {f: _FIELD[f] for f in fields}, 'required': list(fields)}
EXTRACTION_SCHEMA = {'type': 'object', 'additionalProperties': False, 'properties': {'language': {'type': 'string'}, 'findings': {'type': 'object', 'additionalProperties': False, 'properties': {t: _item(TARGET_FIELDS[t]) for t in TARGETS}, 'required': list(TARGETS)}}, 'required': ['language', 'findings']}
_CRITERIA_BLOCK = '\n'.join((f'- {t}: {HOST_CRITERIA[t]}' for t in TARGETS))
SYSTEM_PROMPT = 'You are a musculoskeletal radiologist reading knee MRI reports in many languages. For each of twelve findings you give a calibrated probability that the finding is present IN THE KNEE, together with the evidence and structured detail that justifies it.\n\nYou are NOT merely transcribing. `probability` is your judgement, formed from the whole report: how confidently the finding is stated, the severity words used, what else is described, and what a radiologist would conclude. It is the primary output.\n\nTHE STANDARD YOU ARE JUDGING AGAINST\n\nThese labels were assigned by two subspecialty MSK radiologists reading the IMAGES, adjudicated by a third. Two consequences:\n\n  (a) Their criteria are strict and severity-based. Judge against these definitions,       not against whether the report mentions the structure:\n{criteria}\n\n  (b) They graded ambiguous or borderline findings as NEGATIVE to favour specificity.       When the report is genuinely on the fence, your probability belongs below 0.5.\n\n  (c) They read the images; the report may omit a finding that was present, or state       something the images contradict. So a finding the report never mentions is not       probability 0 - it is the base rate for that finding given a report of this       thoroughness. A terse report that omits synovitis says far less than a detailed       one that lists every structure.\n\nCALIBRATION. Spread your probabilities out; do not snap to 0.05 / 0.5 / 0.95. If you label many findings 0.8, about 80% of them should be truly present.\n\nTHE STRUCTURED FIELDS record what the report literally SAYS, independently of your probability. Fill them faithfully even when your probability disagrees.\n\n- assertion: present / absent / equivocal / not_mentioned.\n- severity: the AMOUNT or SIZE stated (low = trace/small/mild, moderate, high = large/marked/severe). "unqualified" if asserted with no size word.\n- extent: the GRADE of a tear, a different axis from size: low_grade (sprain, grade I), partial, high_grade_partial (>50% fibres), complete (full-thickness, rupture). "unspecified" if a tear is asserted without a grade.\n- chronicity: acute/recent versus chronic/remote/old.\n- surface_contact (menisci): does abnormal signal reach the meniscal surface, or is it intrasubstance/degenerative only?\n- discrete_fracture_line: does the report describe an actual cortical break or fracture line, as opposed to marrow oedema or contusion alone?\n- evidence: a span copied VERBATIM from the report in its original language, the shortest that justifies your fields. Empty string when assertion="not_mentioned".\n\nNegations ("no meniscal tear", "sin derrame") are assertion="absent". Hedges ("possible", "cannot exclude") are assertion="equivocal". Conclusion-stating phrases are assertion="present", NOT equivocal, even where they look tentative in English: "compatible with", "consistent with", Spanish "compatible con", Turkish "ile uyumlu".'
SYSTEM_PROMPT = SYSTEM_PROMPT.replace('{criteria}', _CRITERIA_BLOCK)

def lexicon_block(lexicon: dict, code: str, max_terms: int=12) -> str:
    if not lexicon:
        return ''
    lines = []
    for target, per_language in lexicon.get('findings', {}).items():
        terms = list(per_language.get(code, []))[:max_terms]
        if terms:
            lines.append(f'{target}: {', '.join(terms)}')
    for section, per_language in lexicon.items():
        if section == 'findings':
            continue
        terms = list(per_language.get(code, []))[:max_terms]
        if terms:
            lines.append(f'{section}: {', '.join(terms)}')
    if not lines:
        return ''
    header = f"----- VOCABULARY HINTS, language '{code}' (mined from this corpus; not exhaustive) -----"
    return header + '\n' + '\n'.join(lines)

def build_messages(report: str, lex_block: str) -> list[dict]:
    system = SYSTEM_PROMPT if not lex_block else f'{SYSTEM_PROMPT}\n\n{lex_block}'
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': f'----- REPORT -----\n{report.strip()}\n----- END REPORT -----'}]
LANG_PATTERNS = {'English': '\\b(the|there|no evidence|knee|tear|normal|with)\\b', 'Spanish': '\\b(de la|del|con|sin|rodilla|técnica|resultados)\\b', 'Portuguese': '\\b(joelho|com|sem|não|exame)\\b', 'French': '\\b(du genou|avec|sans|est|le|les)\\b', 'German': '\\b(des|der|und|kein|nicht|Kniegelenk)\\b', 'Italian': '\\b(del|della|con|senza|ginocchio)\\b', 'Turkish': '\\b(ve|ile|yok|diz|izlenmi)', 'Croat/Serb': '\\b(je|se|te|zgloba|koljena|prikaz)\\b', 'Dutch': '\\b(van|het|geen|met|knie)\\b', 'Polish': '\\b(oraz|nie|staw|kolana|widoczne)\\b', 'Russian': '[а-яА-Я]', 'Greek': '[α-ωΑ-Ω]'}
BUCKET_TO_CODE = {'English': 'en', 'Spanish': 'es', 'Portuguese': 'pt', 'French': 'fr', 'German': 'de', 'Italian': 'it', 'Turkish': 'tr', 'Croat/Serb': 'hr', 'Dutch': 'nl', 'Polish': 'pl', 'Russian': 'bg', 'Greek': 'el', 'other': 'en'}

def guess_language(text: str) -> str:
    scores = {k: len(re.findall(p, text, flags=re.I)) for k, p in LANG_PATTERNS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else 'other'

def normalise_report(text: str) -> str:
    return unicodedata.normalize('NFKC', text)


@lru_cache
def load_lexicon(path, expected_sha256):
    content = (Path(__file__).resolve().parents[2] / path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Parag lexicon changed")
    return json.loads(content)


def build_prompt(report, examples, cfg):
    if len(examples) or cfg.request_layout != "joint":
        raise ValueError("prompt_016 uses zero-shot joint requests")
    report = normalise_report(report)
    code = BUCKET_TO_CODE[guess_language(report)]
    lexicon = load_lexicon(cfg.lexicon_json, cfg.lexicon_sha256)
    messages = build_messages(report, lexicon_block(lexicon, code))
    return {
        "system_prompt": messages[0]["content"],
        "prompt": messages[1]["content"],
        "schema": EXTRACTION_SCHEMA,
        "example_ids": [],
        "lexicon_language": code,
    }


def parse_response(response, cfg):
    result = json.loads(response)
    if set(result) != {"language", "findings"} or set(result["findings"]) != set(TARGETS):
        raise ValueError("response keys do not match prompt_016")
    prediction = {"language": result["language"]}
    for target in TARGETS:
        finding = result["findings"][target]
        if set(finding) != set(TARGET_FIELDS[target]):
            raise ValueError(f"invalid finding fields for {target}")
        for field in TARGET_FIELDS[target]:
            value = finding[field]
            if "enum" in _FIELD[field] and value not in _FIELD[field]["enum"]:
                raise ValueError(f"invalid {field} for {target}")
            prediction[f"{field}_{target}"] = value
        probability = finding["probability"]
        if type(probability) not in (int, float) or not 0 <= probability <= 1:
            raise ValueError(f"invalid probability for {target}")
        prediction[f"raw_probability_{target}"] = float(probability)
        prediction[f"score_{target}"] = float(probability)
    return prediction
