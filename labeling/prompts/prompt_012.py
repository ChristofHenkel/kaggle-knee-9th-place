import json

TARGETS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
)
KEYS = (
    "acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa",
    "lateral_oa", "pf_oa", "effusion", "synovitis", "bakers", "contusion", "fracture",
)
PROBABILITIES = tuple(index / 20 for index in range(21))

FINDING_SPEC = """For each finding below, decide what THE REPORT asserts:
- ACL: tear or rupture of the anterior cruciate ligament
- MCL: tear or sprain of the medial collateral ligament
- medial_meniscus: tear of the medial meniscus
- lateral_meniscus: tear of the lateral meniscus
- medial_oa / lateral_oa / pf_oa: osteoarthritis (cartilage loss) in that compartment
- effusion: joint effusion
- synovitis: inflammation or thickening of the synovial lining
- bakers: Baker (popliteal) cyst
- contusion: bone contusion / bone marrow oedema from impact
- fracture: acute fracture or cortical break"""

SYSTEM = """You are an expert musculoskeletal radiologist reading a knee MRI report.
Reports may be written in any language, including Greek, Bulgarian, Turkish, Croatian,
Dutch, German, Spanish, French or English. Read each one in its original language.

For each of the twelve findings below, estimate the probability that the
competition image-based gold annotation is positive, using only evidence in
THE REPORT IN FRONT OF YOU.
{spec}
Interpretation rules:
- Distinguish explicitly affirmed, explicitly denied, suspected/equivocal, and
  not-mentioned findings. "R/O", "possible", and "suspected" are not unequivocal
  positive assertions. Evaluate the actual finding rather than the indication.
- A finding not mentioned is different from one explicitly denied; neither
  establishes its absence from the images.
- Do not equate severity with certainty. An unequivocal mild finding can provide
  stronger evidence of positivity than a suspected severe finding. Do not infer
  moderate severity merely because a finding is present. When a confirmed
  finding has no stated or otherwise supported severity, use minimal severity
  for verdict/severity output rather than inventing a higher grade.
- Contusion: do not automatically classify marrow edema adjacent to degenerative
  changes or an insufficiency fracture as traumatic contusion. A separate
  explicitly described impact-related contusion may coexist with a fracture.
- OA: do not automatically classify an isolated acute osteochondral injury as
  compartmental osteoarthritis. Assess the compartmental cartilage findings.
- Meniscal tear: distinguish a tear from intrasubstance degeneration that does
  not reach a surface and from suspected meniscocapsular injury.
- Resolve conflicting statements using the more definitive Impression/Conclusion
  where it addresses the same finding. Preserve uncertainty if the conflict
  remains unresolved. Do not combine unrelated findings or borrow severity
  from another finding.
- Do not disregard an explicit finding merely because other findings or the
  clinical context make it unexpected.

Probability rules:
The probability expresses confidence in target positivity, not severity.
Retain graded uncertainty for suspected or equivocal findings. Do not promote
an unresolved possibility to the confidence of an unequivocal affirmative.
An explicit report finding is evidence about the images, not a guarantee of the
image-based annotation. Do not invent image findings that the report omits.
Return each probability as exactly one of these 21 levels:
{levels}
Account for explicit positive findings, explicit negation, uncertain statements,
and findings that are not mentioned. Do not treat silence as explicit negation.
Respond with JSON only: one object with all twelve keys, each containing a numeric
probability. Do not output verdicts, severity labels, evidence, or explanations.
""".format(spec=FINDING_SPEC, levels=", ".join(f"{value:.2f}" for value in PROBABILITIES))


def build_prompt(report, examples, cfg):
    if len(examples) or cfg.request_layout != "joint":
        raise ValueError("prompt_012 uses zero-shot joint requests")
    return {
        "prompt": f"{SYSTEM}\nREPORT:\n{report.strip()[:12000]}",
        "schema": {
            "type": "object",
            "properties": {
                key: {"type": "number", "enum": list(PROBABILITIES)} for key in KEYS
            },
            "required": list(KEYS),
            "additionalProperties": False,
        },
        "example_ids": [],
    }


def parse_response(response, cfg):
    result = json.loads(response)
    if not isinstance(result, dict) or set(result) != set(KEYS):
        raise ValueError("response keys do not match prompt_012")
    prediction = {}
    for target, key in zip(TARGETS, KEYS):
        probability = result[key]
        if type(probability) not in (int, float) or probability not in PROBABILITIES:
            raise ValueError(f"invalid 21-level probability for {key}")
        prediction[f"raw_probability_{target}"] = float(probability)
        prediction[f"score_{target}"] = float(probability)
    return prediction
