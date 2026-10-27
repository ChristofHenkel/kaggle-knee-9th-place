import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description='Export prompt requests or parse completed responses; no model calls')
    parser.add_argument('--prompt', choices=['012', '016', '023', '036', '041', '048'], required=True)
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--image-dir', default='outputs/labeling/individual_images')
    parser.add_argument('--responses', help='JSONL records with StudyInstanceUID and response')
    args = parser.parse_args()
    prompt = importlib.import_module('labeling.prompts.prompt_' + args.prompt)
    cfg = SimpleNamespace(input_csv=args.input_csv, image_dir=args.image_dir, request_layout='joint',
                          lexicon_json='labeling/parag_v18_lexicon.json',
                          lexicon_sha256='1be6138ec6caeca7460cb44965f7254e34433e292efd168b6526a8b8f7fc5ad5',
                          host_guide='labeling/host_guide.md',
                          teaching_images=('labeling/host_reference_1.jpg',
                                           'labeling/host_reference_2.jpg'))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.responses:
        rows = []
        for line in Path(args.responses).read_text().splitlines():
            record = json.loads(line)
            response = record['response']
            parsed = prompt.parse_response(response if isinstance(response, str) else json.dumps(response), cfg)
            values = {'StudyInstanceUID': record['StudyInstanceUID']}
            for target in prompt.TARGETS:
                field = 'raw_probability_' + target if 'raw_probability_' + target in parsed else 'score_' + target
                values[target] = parsed[field]
            rows.append(values)
        pd.DataFrame(rows).to_csv(output, index=False)
    else:
        with output.open('w') as stream:
            for row in pd.read_csv(args.input_csv).itertuples(index=False):
                if args.prompt in ('023', '041', '048'):
                    request = prompt.build_requests(row, [], cfg)[0]
                else:
                    request = prompt.build_prompt(row.Report, [], cfg)
                stream.write(json.dumps({'StudyInstanceUID': row.StudyInstanceUID, **request}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
