import sys
from pathlib import Path
if __name__ == '__main__':
    sys.path[0] = str(Path(__file__).resolve().parents[1])
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
import datetime
import hashlib
import io
import json
import re
import sqlite3
import tarfile
import cv2
import numpy as np
import pandas as pd
import pydicom
from utils.prepare_oai import ROOT, cache_series, save_csv, save_json, sha256
TARGETS = ['Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis']

def header_record(task):
    row, root = task
    with tarfile.open(Path(root) / row['archive'], 'r|gz') as archive:
        member = next((member for member in archive if member.isfile()))
        header = pydicom.dcmread(io.BytesIO(archive.extractfile(member).read()), stop_before_pixels=True)
    description = str(header.SeriesDescription).upper()
    side = next((side for token, side in [('RIGHT', 'R'), ('LEFT', 'L')] if token in re.split('[_\\W]+', description)), '')
    selected = str(header.PatientID) == row['OAI_ID'] and str(header.StudyInstanceUID) == row['StudyInstanceUID'] and (side == row['knee_side']) and (str(header.Modality) == 'MR')
    orientation = np.asarray(header.ImageOrientationPatient, float)
    normal = np.cross(orientation[:3], orientation[3:])
    return {'archive': row['archive'], 'patient_id': row['OAI_ID'], 'laterality': row['knee_side'], 'case_id': row['StudyInstanceUID'], 'visit_month': int(row['visit_month']), 'selected': selected, 'actual_study_uid': str(header.StudyInstanceUID), 'actual_patient_id': str(header.PatientID), 'actual_side': side, 'series_uid': str(header.SeriesInstanceUID), 'description': description, 'plane': ('SAG', 'COR', 'AXI')[int(np.abs(normal).argmax())], 'spacing_mm': float(np.mean(np.asarray(header.PixelSpacing, float)))}

def prepare_cache(cfg):
    if sha256(cfg.oai_selected_archives) != cfg.oai_selected_archives_sha256:
        raise ValueError('Pinned OAI download selection differs')
    selection = pd.read_csv(cfg.oai_selected_archives, dtype={'OAI_ID': str})
    frame = pd.read_csv(cfg.train_df, dtype={'patient_id': str})
    selection = selection.loc[selection.StudyInstanceUID.isin(frame.case_id)]
    destination = Path(cfg.cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(destination / 'index.sqlite') as database:
        database.execute('CREATE TABLE IF NOT EXISTS series (archive TEXT PRIMARY KEY, record TEXT NOT NULL)')
        indexed = {row[0] for row in database.execute('SELECT archive FROM series')}
        pending = selection.loc[~selection.archive.isin(indexed)]
        with ThreadPoolExecutor(max_workers=cfg.cache_workers) as pool:
            tasks = [(row, cfg.oai_root) for row in pending.to_dict('records')]
            for count, record in enumerate(pool.map(header_record, tasks), 1):
                database.execute('INSERT INTO series VALUES (?, ?)', (record['archive'], json.dumps(record)))
                if count % 100 == 0:
                    database.commit()
                if count % 1000 == 0 or count == len(tasks):
                    print(f'Labelled OAI header verification: {count}/{len(tasks)}', flush=True)
        database.commit()
        records = [json.loads(row[0]) for row in database.execute('SELECT record FROM series ORDER BY archive')]
    rejected = [row for row in records if not row['selected']]
    save_json(destination / 'identity_rejections.json', rejected)
    selected = pd.DataFrame([row for row in records if row['selected']]).drop_duplicates(['case_id', 'series_uid'])
    missing = set(frame.case_id) - set(selected.case_id)
    if missing:
        raise ValueError(f'{len(missing)} labelled studies have no DICOM series with verified UID, patient and side')
    save_csv(destination / 'indexed_series.csv', selected)
    cached, pending = ([], selected.to_dict('records'))
    values = {key: getattr(cfg, key) for key in ('oai_root', 'cache_dir', 'cache_slices', 'img_px', 'crop_mm', 'pct_low', 'pct_high')}
    with ProcessPoolExecutor(max_workers=cfg.cache_workers, initializer=cv2.setNumThreads, initargs=(0,)) as pool:
        for count, record in enumerate(pool.map(cache_series, [(row, values) for row in pending], chunksize=1), 1):
            cached.append(record)
            if count % 100 == 0 or count == len(pending):
                print(f'Labelled OAI cache: {count}/{len(pending)} new volumes', flush=True)
    result = pd.DataFrame(cached).sort_values(['case_id', 'series_uid']).reset_index(drop=True)
    save_csv(cfg.oai_series_csv, result)
    print(f'Labelled OAI cache ready: {len(result)} volumes, {result.case_id.nunique()} studies', flush=True)
    return result
