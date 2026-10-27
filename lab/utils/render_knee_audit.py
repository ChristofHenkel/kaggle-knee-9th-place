import argparse
import hashlib
import io
import json
import shutil
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT / 'assets/data/frozen_training_metadata.csv'
PLANES = [('Sagittal', 'SAG'), ('Coronal', 'COR'), ('Axial', 'AXI')]


def sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def direction(vector):
    letters = [('R', 'L'), ('A', 'P'), ('I', 'S')]
    return ''.join(letters[i][int(vector[i] > 0)] for i in np.argsort(-np.abs(vector)) if abs(vector[i]) >= .2)


class KneeAuditRenderer:
    def __init__(self, root=ROOT, metadata_csv=None, tile_pixels=320, fov_mm=160, overlay_training_crop=False, excluded_series=()):
        self.root = Path(root)
        self.tile_pixels = tile_pixels
        self.fov_mm = fov_mm
        self.overlay = overlay_training_crop
        metadata = Path(metadata_csv) if metadata_csv is not None else self.root / METADATA.relative_to(ROOT)
        self.metadata_hashes = {metadata.name: sha256(metadata)}
        self.archive_hash = self.metadata_hashes[metadata.name]
        frozen = pd.read_csv(metadata)
        self.centers = frozen[['SeriesInstanceUID', 'coordinate_status', 'center_pixel_column', 'center_pixel_row']].set_index('SeriesInstanceUID')
        self.spacing = frozen[['SeriesInstanceUID', 'PixelSpacingRow', 'PixelSpacingColumn']].set_index('SeriesInstanceUID')
        self.laterality = frozen.loc[frozen.predicted_laterality.notna(), ['StudyInstanceUID', 'SeriesInstanceUID', 'position_axis', 'position_sign', 'predicted_laterality']].set_index('SeriesInstanceUID')
        self.selected = pd.read_csv(self.root / 'assets/data/selected_slots_1_85.csv')
        self.series = pd.read_csv(self.root / 'assets/data/rsna-knee-abnormality-detection/train_series.csv')
        if excluded_series:
            self.series = self.series.loc[~self.series.SeriesInstanceUID.isin(excluded_series)]
            self.selected = self.selected.loc[~self.selected.SeriesInstanceUID.isin(excluded_series)]
        cache = self.root / 'assets/data/parag'
        converted = pd.read_csv(cache / 'train_series_npz_uint16_metadata.csv')
        self.cache_metadata = converted.set_index('SeriesInstanceUID')
        self.font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
        self.header_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)

    def choose_series(self, uid):
        available = self.series[self.series.StudyInstanceUID == uid]
        selected = self.selected[self.selected.StudyInstanceUID == uid]
        panels, missing = [], []
        for plane, short in PLANES:
            candidates = available[available.Anatomical_Plane == plane].copy()
            if candidates.empty:
                missing.append({'plane': plane, 'reason': 'no acquired series'})
                continue
            preferred = selected[selected.slot == short + '_FS']
            if len(preferred):
                series_uid = preferred.iloc[0].SeriesInstanceUID
            else:
                candidates = candidates.sort_values(['Fluid_Sensitive', 'Fat_Suppression', 'SeriesInstanceUID'], ascending=[False, False, True])
                series_uid = candidates.iloc[0].SeriesInstanceUID
            row = available[available.SeriesInstanceUID == series_uid].iloc[0]
            panels.append((row, False, {'requested': short + '_FS', 'fallback': not bool(row.Fluid_Sensitive)}))
        nfs = selected[selected.slot.str.startswith('SAG_NFS')]
        if len(nfs):
            series_uid = nfs.sort_values('slot').iloc[0].SeriesInstanceUID
            detail = available[available.SeriesInstanceUID == series_uid].iloc[0]
        else:
            options = available[(available.Anatomical_Plane == 'Sagittal') & (available.Fluid_Sensitive == 0)]
            if len(options):
                detail = options.sort_values('SeriesInstanceUID').iloc[0]
            elif len(panels):
                detail = panels[0][0]
            else:
                raise ValueError(f'No available imaging series for {uid}')
        panels.append((detail, True, {'requested': 'SAG_NFS_DETAIL', 'fallback': not (detail.Anatomical_Plane == 'Sagittal' and detail.Fluid_Sensitive == 0)}))
        return panels, missing

    def load_series(self, uid, row):
        series_uid = row.SeriesInstanceUID
        path = self.root / 'assets/data/parag/train_series_npz_uint16' / uid / (series_uid + '.npz')
        with np.load(path) as data:
            volume = data['volume']
        meta = self.cache_metadata.loc[series_uid]
        source = self.root / 'assets/data/rsna-knee-abnormality-detection/train_series' / uid / series_uid
        skipped = str(meta.get('skipped_sop_instance_uids', '')).split(';')
        tags = ['Rows', 'Columns', 'PixelSpacing', 'ImageOrientationPatient', 'ImagePositionPatient', 'SOPInstanceUID', 'InstanceNumber', 'SeriesDescription']
        headers = [pydicom.dcmread(p, stop_before_pixels=True, specific_tags=tags) for p in source.glob('*.dcm') if p.stem not in skipped]
        h = headers[0]
        iop = np.array(h.ImageOrientationPatient, dtype=float)
        spacing = np.array(h.PixelSpacing, dtype=float)
        if volume.shape != (int(meta.output_n_slices), int(h.Rows), int(h.Columns)) or len(headers) != volume.shape[0]:
            raise ValueError(f'Cache/DICOM dimensions mismatch for {series_uid}')
        frozen_spacing = self.spacing.loc[series_uid]
        if not np.allclose(spacing, [frozen_spacing.PixelSpacingRow, frozen_spacing.PixelSpacingColumn], atol=1e-5):
            raise ValueError(f'Frozen/DICOM spacing mismatch for {series_uid}')
        for header in headers:
            if not np.allclose(header.ImageOrientationPatient, iop, atol=1e-4) or not np.allclose(header.PixelSpacing, spacing, atol=1e-5):
                raise ValueError(f'Inconsistent DICOM geometry for {series_uid}')
        normal = np.cross(iop[:3], iop[3:])
        if meta.order_method == 'image_position_patient_normal':
            headers.sort(key=lambda d: float(np.dot(d.ImagePositionPatient, normal)))
        elif meta.order_method == 'instance_number':
            headers.sort(key=lambda d: int(d.InstanceNumber))
        else:
            raise ValueError(f'Unsupported cache order method {meta.order_method}')
        center = self.centers.loc[series_uid]
        if center.coordinate_status != 'ok':
            raise ValueError(f'Unavailable frozen center for {series_uid}')
        return volume, spacing, iop, headers, path, [float(center.center_pixel_column), float(center.center_pixel_row)], meta

    def series_catalog(self, uid):
        available = self.series[self.series.StudyInstanceUID == uid].sort_values('SeriesInstanceUID')
        catalog = []
        for number, (_, row) in enumerate(available.iterrows()):
            series_uid = row.SeriesInstanceUID
            path = self.root / 'assets/data/parag/train_series_npz_uint16' / uid / (series_uid + '.npz')
            if not path.is_file():
                continue
            _, spacing, iop, headers, _, center, meta = self.load_series(uid, row)
            catalog.append({'series_key': f'S{number:02d}', 'plane': row.Anatomical_Plane,
                            'fluid_sensitive': int(row.Fluid_Sensitive), 'fat_suppression': int(row.Fat_Suppression),
                            'series_description': str(getattr(headers[0], 'SeriesDescription', '')),
                            'shape': [int(meta.output_n_slices), int(meta.Rows), int(meta.Columns)],
                            'center_column_row': center, 'spacing_row_column': spacing.tolist(), 'iop': iop.tolist(),
                            'slice_index_direction': direction(np.cross(iop[:3], iop[3:])) if meta.order_method == 'image_position_patient_normal' else 'unknown'})
        return catalog

    def render_tile_montage(self, uid, selection, output_dir, filename='04_SELECTED_DETAIL.jpg'):
        start = time.monotonic()
        tiles = selection.get('tiles')
        if not isinstance(tiles, list) or not 1 <= len(tiles) <= 16:
            raise ValueError('Selection must contain between one and sixteen tiles')
        if not isinstance(selection.get('selection_reason'), str):
            raise ValueError('Selection reason must be a string')
        available = self.series[self.series.StudyInstanceUID == uid].sort_values('SeriesInstanceUID')
        rows = {f'S{number:02d}': row for number, (_, row) in enumerate(available.iterrows())}
        output_dir = Path(output_dir) / uid
        if Path(filename).name != filename or not filename.lower().endswith('.jpg'):
            raise ValueError('Montage filename must be a JPEG basename')
        loaded, observations = {}, []
        tile = self.tile_pixels
        canvas = Image.new('RGB', (4 * tile, 4 * (tile + 24) + 48), 'black')
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), 'Agent-selected native detail slices | individual crops and sequences', font=self.header_font, fill='white')
        for cell, choice in enumerate(tiles):
            key = choice.get('series_key')
            if key not in rows:
                raise ValueError(f'Unknown series key {key}')
            if key not in loaded:
                row = rows[key]
                volume, spacing, iop, headers, source, center, meta = self.load_series(uid, row)
                window = np.percentile(volume, [1, 99])
                if window[1] <= window[0]:
                    raise ValueError(f'Empty intensity window for {key}')
                loaded[key] = (row, volume, spacing, iop, headers, source, window, sha256(source))
            row, volume, spacing, iop, headers, source, window, source_hash = loaded[key]
            index = choice.get('index')
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(volume):
                raise ValueError(f'Invalid native slice index for {key}: {index}')
            center_column, center_row, fov = (choice.get(k) for k in ['center_column', 'center_row', 'fov_mm'])
            values = [center_column, center_row, fov]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in values):
                raise ValueError(f'Invalid crop coordinates or FOV for {key}')
            depth, height, width = volume.shape
            if not (0 <= center_column <= width - 1 and 0 <= center_row <= height - 1 and 40 <= fov <= 180):
                raise ValueError(f'Crop center must be in bounds and FOV between40 and180mm for {key}')
            want_h, want_w = np.round(fov / spacing).astype(int)
            x0, y0 = int(round(center_column - want_w / 2)), int(round(center_row - want_h / 2))
            crop = np.zeros((want_h, want_w), dtype=np.uint16)
            sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(width, x0 + want_w), min(height, y0 + want_h)
            crop[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = volume[index, sy0:sy1, sx0:sx1]
            pixels = np.rint(np.clip((crop.astype(np.float32) - window[0]) / (window[1] - window[0]), 0, 1) * 255).astype(np.uint8)
            resized = cv2.resize(pixels, (tile, tile), interpolation=cv2.INTER_AREA if max(want_h, want_w) > tile else cv2.INTER_LINEAR)
            x, y = cell % 4 * tile, 48 + cell // 4 * (tile + 24)
            canvas.paste(Image.fromarray(resized).convert('RGB'), (x, y))
            short = dict(PLANES)[row.Anatomical_Plane]
            caption = f'{key} {short} {"FS" if row.Fluid_Sensitive else "NFS"} z{index} FOV{fov:g}mm'
            draw.text((x + 4, y + tile + 2), caption, font=self.font, fill='white')
            for mx, my, marker in [(x + 4, y + tile // 2, direction(-iop[:3])), (x + tile - 26, y + tile // 2, direction(iop[:3])), (x + tile // 2, y + 3, direction(-iop[3:])), (x + tile // 2, y + tile - 22, direction(iop[3:]))]:
                draw.text((mx, my), marker, font=self.font, fill='#ffee88', stroke_width=1, stroke_fill='black')
            observations.append({'series_key': key, 'SeriesInstanceUID': row.SeriesInstanceUID, 'plane': row.Anatomical_Plane,
                                 'fluid_sensitive': int(row.Fluid_Sensitive), 'fat_suppression': int(row.Fat_Suppression),
                                 'series_description': str(getattr(headers[0], 'SeriesDescription', '')), 'shape': list(volume.shape),
                                 'index': index, 'center_column_row': [center_column, center_row], 'fov_mm': fov,
                                 'spacing_row_column': spacing.tolist(), 'iop': iop.tolist(), 'window': window.tolist(),
                                 'display_crop_native_xywh': [x0, y0, int(want_w), int(want_h)], 'caption': caption,
                                 'source_path': str(source), 'source_sha256': source_hash,
                                 'SOPInstanceUID': str(headers[index].SOPInstanceUID),
                                 'position_patient': list(map(float, headers[index].ImagePositionPatient))})
        output_dir.mkdir(parents=True, exist_ok=True)
        canvas.save(output_dir / filename, quality=93)
        serialized_selection = json.dumps(selection, sort_keys=True, separators=(',', ':'))
        panel = {'file': filename, 'plane': 'Mixed', 'fluid_sensitive': -1, 'fat_suppression': -1,
                 'series_description': 'Agent-selected native tiles', 'shape': None,
                 'indices': [item['index'] for item in observations], 'consecutive': False,
                 'spacing_row_column': None, 'iop': None, 'tiles': observations,
                 'selection_sha256': hashlib.sha256(serialized_selection.encode()).hexdigest(),
                 'selection_reason': selection['selection_reason'], 'elapsed_seconds': time.monotonic() - start,
                 'image_sha256': sha256(output_dir / filename)}
        (output_dir / (Path(filename).stem + '_selection.json')).write_text(json.dumps(selection, indent=2) + '\n')
        (output_dir / (Path(filename).stem + '_metadata.json')).write_text(json.dumps(panel, indent=2) + '\n')
        return panel

    def render_selected_detail(self, uid, selection, output_dir):
        panel = self.render_tile_montage(uid, selection, output_dir, '04_SELECTED_DETAIL.jpg')
        output_dir = Path(output_dir) / uid
        fixed_dir = self.root / 'outputs/labeling/vision_gt58_inputs/images' / uid
        original = json.loads((fixed_dir / 'manifest.json').read_text())
        if len(original['panels']) < 3:
            raise ValueError('Adaptive review requires three overview panels')
        for overview in original['panels'][:3]:
            shutil.copy2(fixed_dir / overview['file'], output_dir / overview['file'])
        manifest = dict(original)
        manifest['panels'] = original['panels'][:3] + [panel]
        manifest['selection_sha256'] = panel['selection_sha256']
        manifest['selection_reason'] = panel['selection_reason']
        manifest['elapsed_seconds'] = panel['elapsed_seconds']
        manifest['overview_image_sha256'] = {item['file']: sha256(output_dir / item['file']) for item in original['panels'][:3]}
        (output_dir / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
        (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        return manifest

    def create_navigation_case(self, uid, output_dir, fixed_image_dir=None):
        output_dir = Path(output_dir) / uid
        output_dir.mkdir(parents=True, exist_ok=True)
        fixed_dir = Path(fixed_image_dir) / uid if fixed_image_dir is not None else self.root / 'outputs/labeling/vision_gt58_inputs/images' / uid
        original = json.loads((fixed_dir / 'manifest.json').read_text())
        catalog_file = self.root / 'outputs/labeling/vision_gt58_inputs/catalogs' / (uid + '.json')
        catalog = json.loads(catalog_file.read_text()) if catalog_file.is_file() else {'series': self.series_catalog(uid)}
        rows = self.series[self.series.StudyInstanceUID == uid].sort_values('SeriesInstanceUID')
        aliases = {row.SeriesInstanceUID: f'S{number:02d}' for number, (_, row) in enumerate(rows.iterrows())}
        tile = 256
        canvas = Image.new('RGB', (6 * tile, 4 * (tile + 24) + 48), 'black')
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), 'Native MRI overview | 8 slices per plane | select further native slices using series catalog', font=self.header_font, fill='white')
        observations = []
        for panel in original['panels'][:3]:
            source_image = Image.open(fixed_dir / panel['file']).convert('RGB')
            n = len(panel['indices'])
            cells = np.unique(np.round(np.linspace(0, n - 1, min(8, n))).astype(int))
            for source_cell in cells:
                cell = len(observations)
                sx, sy = int(source_cell % 4 * self.tile_pixels), int(48 + source_cell // 4 * (self.tile_pixels + 24))
                crop = np.asarray(source_image.crop((sx, sy, sx + self.tile_pixels, sy + self.tile_pixels)))
                resized = cv2.resize(crop, (tile, tile), interpolation=cv2.INTER_AREA)
                x, y = cell % 6 * tile, 48 + cell // 6 * (tile + 24)
                canvas.paste(Image.fromarray(resized), (x, y))
                key = aliases[panel['SeriesInstanceUID']]
                index = panel['indices'][int(source_cell)]
                caption = f'{key} {dict(PLANES)[panel["plane"]]} {"FS" if panel["fluid_sensitive"] else "NFS"} z{index}'
                draw.text((x + 4, y + tile + 2), caption, font=self.font, fill='white')
                observations.append({'series_key': key, 'index': index, 'plane': panel['plane'],
                                     'fluid_sensitive': panel['fluid_sensitive'], 'fat_suppression': panel['fat_suppression'],
                                     'series_description': panel['series_description'], 'shape': panel['shape'],
                                     'center_column_row': panel['center_column_row'], 'fov_mm': original['fov_mm'],
                                     'spacing_row_column': panel['spacing_row_column'], 'iop': panel['iop'],
                                     'source_overview_file': str(fixed_dir / panel['file']),
                                     'source_overview_sha256': sha256(fixed_dir / panel['file']),
                                     'source_overview_cell': int(source_cell), 'native_reference': panel['slices'][int(source_cell)]})
        canvas.save(output_dir / '01_OVERVIEW.jpg', quality=93)
        panel = {'file': '01_OVERVIEW.jpg', 'plane': 'Mixed', 'fluid_sensitive': -1, 'fat_suppression': -1,
                 'series_description': 'Three-plane overview', 'shape': None, 'indices': [v['index'] for v in observations],
                 'consecutive': False, 'spacing_row_column': None, 'iop': None, 'tiles': observations,
                 'image_sha256': sha256(output_dir / '01_OVERVIEW.jpg')}
        manifest = {'StudyInstanceUID': uid, 'geometry_identity_sha256': self.archive_hash,
                    'geometry_metadata_sha256': self.metadata_hashes, 'depth_resampling': False,
                    'tile_pixels': tile, 'frozen_laterality_predictions': original['frozen_laterality_predictions'], 'panels': [panel]}
        (output_dir / 'catalog.json').write_text(json.dumps(catalog, indent=2) + '\n')
        (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        return manifest

    def render_study(self, uid, output_dir):
        start = time.monotonic()
        output_dir = Path(output_dir) / uid
        output_dir.mkdir(parents=True, exist_ok=True)
        chosen, missing = self.choose_series(uid)
        manifest = {'StudyInstanceUID': uid, 'geometry_identity_sha256': self.archive_hash, 'geometry_metadata_sha256': self.metadata_hashes, 'tile_pixels': self.tile_pixels, 'fov_mm': self.fov_mm, 'normalization_percentiles': [1, 99], 'depth_resampling': False, 'training_crop_overlay': self.overlay, 'missing_planes': missing, 'frozen_laterality_predictions': sorted(self.laterality[self.laterality.StudyInstanceUID == uid].predicted_laterality.dropna().unique().tolist()), 'panels': []}
        for number, (row, consecutive, choice) in enumerate(chosen, 1):
            volume, spacing, iop, headers, source_path, center, meta = self.load_series(uid, row)
            depth, height, width = volume.shape
            if consecutive:
                first = max(0, (depth - 16) // 2)
                indices = np.arange(first, min(depth, first + 16))
            else:
                indices = np.unique(np.round(np.linspace(0, depth - 1, min(16, depth))).astype(int))
            window = np.percentile(volume, [1, 99])
            if window[1] <= window[0]:
                raise ValueError(f'Empty intensity window for {row.SeriesInstanceUID}')
            want_h, want_w = np.round(self.fov_mm / spacing).astype(int)
            x0, y0 = int(round(center[0] - want_w / 2)), int(round(center[1] - want_h / 2))
            training_want = int(round(110 / float(np.mean([meta.PixelSpacing_row, meta.PixelSpacing_col]))))
            training_crop = [width // 2 - training_want // 2, height // 2 - training_want // 2, training_want, training_want]
            short = dict(PLANES)[row.Anatomical_Plane]
            label = f'{short} FS={int(row.Fluid_Sensitive)} FAT_SUP={int(row.Fat_Suppression)}'
            name = f'{number:02d}_{short}_{"FS" if row.Fluid_Sensitive else "NFS"}{"_DETAIL" if consecutive else ""}.jpg'
            tile = self.tile_pixels
            canvas = Image.new('RGB', (4 * tile, 4 * (tile + 24) + 48), 'black')
            draw = ImageDraw.Draw(canvas)
            description = str(getattr(headers[0], 'SeriesDescription', ''))
            draw.text((8, 8), f'{label} | {"consecutive detail" if consecutive else "full-depth overview"} | {description}', font=self.header_font, fill='white')
            for cell, index in enumerate(indices):
                native = volume[index]
                crop = np.zeros((want_h, want_w), dtype=np.uint16)
                sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(width, x0 + want_w), min(height, y0 + want_h)
                crop[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = native[sy0:sy1, sx0:sx1]
                pixels = np.rint(np.clip((crop.astype(np.float32) - window[0]) / (window[1] - window[0]), 0, 1) * 255).astype(np.uint8)
                interp = cv2.INTER_AREA if max(want_h, want_w) > tile else cv2.INTER_LINEAR
                resized = cv2.resize(pixels, (tile, tile), interpolation=interp)
                x, y = cell % 4 * tile, 48 + cell // 4 * (tile + 24)
                canvas.paste(Image.fromarray(resized).convert('RGB'), (x, y))
                draw.text((x + 5, y + tile + 2), f'slice {int(index)} / {depth-1}', font=self.font, fill='white')
                markers = [(x + 4, y + tile // 2, direction(-iop[:3])), (x + tile - 26, y + tile // 2, direction(iop[:3])), (x + tile // 2, y + 3, direction(-iop[3:])), (x + tile // 2, y + tile - 22, direction(iop[3:]))]
                for mx, my, text in markers:
                    draw.text((mx, my), text, font=self.font, fill='#ffee88', stroke_width=1, stroke_fill='black')
                if self.overlay:
                    tx, ty, tw, th = training_crop
                    draw.rectangle((x + (tx-x0)/want_w*tile, y + (ty-y0)/want_h*tile, x + (tx+tw-x0)/want_w*tile, y + (ty+th-y0)/want_h*tile), outline='#00dddd', width=2)
            canvas.save(output_dir / name, quality=93)
            laterality = self.laterality.loc[row.SeriesInstanceUID].to_dict() if row.SeriesInstanceUID in self.laterality.index else None
            manifest['panels'].append({'file': name, 'SeriesInstanceUID': row.SeriesInstanceUID, 'plane': row.Anatomical_Plane, 'fluid_sensitive': int(row.Fluid_Sensitive), 'fat_suppression': int(row.Fat_Suppression), 'series_description': description, 'shape': list(volume.shape), 'indices': indices.tolist(), 'consecutive': consecutive, 'center_column_row': center, 'spacing_row_column': spacing.tolist(), 'iop': iop.tolist(), 'source_path': str(source_path), 'source_dicom_directory': str(self.root / 'assets/data/rsna-knee-abnormality-detection/train_series' / uid / row.SeriesInstanceUID), 'source_sha256': sha256(source_path), 'order_method': meta.order_method, 'window': window.tolist(), 'display_crop_native_xywh': [x0, y0, int(want_w), int(want_h)], 'training_crop_native_xywh': training_crop, 'laterality': laterality, 'choice': choice, 'slices': [{'index': int(i), 'SOPInstanceUID': str(headers[i].SOPInstanceUID), 'position_patient': list(map(float, headers[i].ImagePositionPatient))} for i in indices]})
        manifest['elapsed_seconds'] = time.monotonic() - start
        (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--overlay-training-crop', action='store_true')
    parser.add_argument('--navigation', action='store_true')
    parser.add_argument('--selection')
    parser.add_argument('--filename', default='02_NAVIGATION.jpg')
    args = parser.parse_args()
    studies = pd.read_csv(args.input_csv)
    if set(studies.columns) - {'StudyInstanceUID', 'Report'}:
        raise ValueError('Input CSV must contain only study identifiers and optional reports')
    from lab.utils.prepare_knee import REPAIRS
    renderer = KneeAuditRenderer(overlay_training_crop=args.overlay_training_crop, excluded_series=tuple(REPAIRS))
    for uid in studies.StudyInstanceUID:
        if args.selection:
            panel = renderer.render_tile_montage(uid, json.loads(Path(args.selection).read_text()), args.output_dir, args.filename)
            path = Path(args.output_dir) / uid / 'manifest.json'
            manifest = json.loads(path.read_text())
            manifest['panels'].append(panel)
            path.write_text(json.dumps(manifest, indent=2) + '\n')
        elif args.navigation:
            fixed = Path(args.output_dir) / 'fixed'
            renderer.render_study(uid, fixed)
            manifest = renderer.create_navigation_case(uid, args.output_dir, fixed)
            context = {'StudyInstanceUID': uid, 'catalog': str(Path(args.output_dir) / uid / 'catalog.json')}
            (Path(args.output_dir) / uid / 'audit_context.json').write_text(json.dumps(context, indent=2) + '\n')
        else:
            manifest = renderer.render_study(uid, args.output_dir)
        print(json.dumps({'StudyInstanceUID': uid, 'panels': len(manifest['panels']), 'missing_planes': manifest['missing_planes'], 'elapsed_seconds': round(manifest.get('elapsed_seconds', 0), 2)}), flush=True)


if __name__ == '__main__':
    main()
