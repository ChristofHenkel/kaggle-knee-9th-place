import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset
from ds_knee import SLOTS, TARGETS, PLANE_CODE, derive_series_weight, physical_crop_resize, slice_plan, crop_study_images
from utils.prepare_knee import REPAIRS
HEADER_TAGS = ('Laterality', 'ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing', 'Rows', 'Columns', 'InstanceNumber', 'SeriesDescription', 'SequenceName', 'ScanningSequence', 'RepetitionTime', 'EchoTime', 'SliceThickness')

def mode(values):
    values = [v for v in values if v is not None]
    return pd.Series(values).mode().iloc[0] if values else None

def center_x(header):
    required = ('ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing', 'Rows', 'Columns')
    if any((header.get(k) is None for k in required)):
        return None
    try:
        position = np.asarray(header['ImagePositionPatient'], dtype=np.float64)
        orientation = np.asarray(header['ImageOrientationPatient'], dtype=np.float64)
        spacing = np.asarray(header['PixelSpacing'], dtype=np.float64)
        rows, columns = (int(header['Rows']), int(header['Columns']))
    except (TypeError, ValueError):
        return None
    if position.size < 3 or orientation.size < 6 or spacing.size < 2 or (min(spacing) <= 0):
        return None
    center = position[:3] + orientation[:3] * spacing[1] * (columns - 1) / 2
    center += orientation[3:6] * spacing[0] * (rows - 1) / 2
    return float(center[0]) if np.isfinite(center[0]) else None

def series_laterality(headers):
    tags = []
    for h in headers:
        value = str(h.get('Laterality') or '').strip().upper()
        value = {'LEFT': 'L', 'RIGHT': 'R'}.get(value, value)
        if value in ('L', 'R'):
            tags.append(value)
    centers = [v for h in headers if (v := center_x(h)) is not None]
    center = float(np.median(centers)) if centers else None
    if tags:
        return {'laterality': mode(tags), 'source': 'dicom_tag', 'center_x': center}
    if center is None or abs(center) <= 10:
        return {'laterality': None, 'source': 'unresolved', 'center_x': center}
    return {'laterality': 'R' if center < 0 else 'L', 'source': 'geometry', 'center_x': center}

def study_laterality(records):
    tagged = [r['laterality'] for r in records if r['source'] == 'dicom_tag']
    centers = [r['center_x'] for r in records if r['center_x'] is not None]
    center = float(np.median(centers)) if centers else None
    if tagged:
        return mode(tagged)
    return ('R' if center < 0 else 'L') if center is not None and abs(center) > 10 else None

def acquisition_meta(headers):
    result = {}
    for tag in ('SeriesDescription', 'SequenceName', 'ScanningSequence'):
        values = [str(h[tag]) for h in headers if h.get(tag) is not None and str(h[tag])]
        result[tag + '_mode'] = mode(values)
    for tag in ('RepetitionTime', 'EchoTime'):
        values = pd.to_numeric(pd.Series([h.get(tag) for h in headers]), errors='coerce')
        values = values[np.isfinite(values)]
        result[tag + '_median'] = float(values.median()) if len(values) else np.nan
    return result

def scan_series(directory):
    paths = sorted(Path(directory).glob('*.dcm'))
    headers = []
    geometry = []
    for path in paths:
        header = {}
        try:
            ds = pydicom.dcmread(path, force=True, stop_before_pixels=True)
        except Exception:
            ds = None
        for tag in HEADER_TAGS:
            try:
                header[tag] = getattr(ds, tag, None)
            except Exception:
                header[tag] = None
        headers.append(header)
        try:
            iop = np.asarray(header['ImageOrientationPatient'], dtype=np.float64)
            ipp = np.asarray(header['ImagePositionPatient'], dtype=np.float64)
            if iop.size < 6 or ipp.size < 3:
                raise ValueError('missing geometry')
            projection = float(np.dot(ipp[:3], np.cross(iop[:3], iop[3:6])))
            instance = float(header['InstanceNumber'] if header['InstanceNumber'] is not None else np.inf)
        except (TypeError, ValueError):
            projection, instance = (np.nan, np.inf)
        geometry.append((path, projection, instance))
    need = max(2, int(0.8 * len(paths)))
    if sum((np.isfinite(r[1]) for r in geometry)) >= need:
        ordered = sorted(geometry, key=lambda r: (r[1] if np.isfinite(r[1]) else np.inf, r[2], str(r[0])))
        order = 'image_position_patient_normal'
    elif sum((np.isfinite(r[2]) for r in geometry)) >= need:
        ordered = sorted(geometry, key=lambda r: (r[2], str(r[0])))
        order = 'instance_number'
    else:
        ordered = sorted(geometry, key=lambda r: str(r[0]))
        order = 'filename_fallback'
    ordered = [r[0] for r in ordered if r[0].stem != REPAIRS.get(Path(directory).name)]
    first = headers[paths.index(ordered[0])] if ordered else {}
    spacing = first.get('PixelSpacing')
    try:
        spacing = float(np.mean([float(v) for v in spacing[:2]])) if spacing is not None else np.nan
    except (TypeError, ValueError):
        spacing = np.nan
    try:
        thickness = float(first.get('SliceThickness'))
    except (TypeError, ValueError):
        thickness = 3.0
    thickness = float(np.clip(thickness, 0.5, 8.0)) if np.isfinite(thickness) else 3.0
    return {'paths': ordered, 'n_slices': len(ordered), 'spacing_mm': spacing, 'z_mm': thickness, 'order_method': order, 'laterality': series_laterality(headers), **acquisition_meta(headers)}

def scan_study(uid, series, image_root, extra_slots=0, use_acquisition_metadata=True):
    records = []
    scans = {}
    for row in series.itertuples(index=False):
        scan = scan_series(Path(image_root) / uid / row.SeriesInstanceUID)
        scans[row.SeriesInstanceUID] = scan
        records.append({'StudyInstanceUID': uid, 'SeriesInstanceUID': row.SeriesInstanceUID, 'plane': PLANE_CODE.get(row.Anatomical_Plane), 'Fat_Suppression': row.Fat_Suppression, 'Fluid_Sensitive': row.Fluid_Sensitive, **{k: v for k, v in scan.items() if k not in ('paths', 'laterality')}})
    frame = pd.DataFrame(records)
    laterality = study_laterality([s['laterality'] for s in scans.values()])
    if frame.empty:
        return (frame, scans, laterality)
    frame = frame.loc[frame.plane.notna() & frame.n_slices.gt(0)].copy()
    fs = frame.Fat_Suppression.astype(int).eq(1)
    fluid = derive_series_weight(frame) if use_acquisition_metadata else frame.Fluid_Sensitive.astype(int).eq(1)
    split = frame.plane.isin(('COR', 'SAG')) & ~fs
    frame['slot'] = frame.plane + np.where(fs, '_FS', np.where(split, np.where(fluid, '_NFS_FL', '_NFS_ST'), '_NFS'))
    frame = frame.sort_values(['StudyInstanceUID', 'slot', 'n_slices'], ascending=[True, True, False])
    selected = frame.groupby(['StudyInstanceUID', 'slot'], as_index=False).first()
    if extra_slots:
        selected['slot_id'] = selected.slot.map({slot: index for index, slot in enumerate(SLOTS)})
        rest = frame.loc[~frame.SeriesInstanceUID.isin(selected.SeriesInstanceUID)].copy()
        rest = rest.sort_values(['StudyInstanceUID', 'n_slices', 'SeriesInstanceUID'], ascending=[True, False, True]).head(extra_slots)
        rest['slot_id'] = np.arange(len(rest)) + len(SLOTS)
        selected = pd.concat([selected, rest], ignore_index=True)
    return (selected, scans, laterality)

def volume_to_slot(cfg, volume, spacing, slot, laterality):
    if getattr(cfg, 'normalize_laterality', True) and laterality == 'R':
        volume = volume[::-1] if slot.startswith('SAG') else [sl[:, ::-1] for sl in volume]
    indices, real = slice_plan(len(volume), cfg.cache_slices)
    sub = [volume[i].astype(np.float32) for i in indices]
    pixels = np.concatenate([sl.ravel() for sl, valid in zip(sub, real) if valid > 0])
    lo, hi = np.percentile(pixels, [cfg.pct_low, cfg.pct_high])
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1
    lo, span = (np.float32(lo), np.float32(hi - lo))
    images = np.stack([physical_crop_resize(cfg, np.clip((sl - lo) / span, 0, 1) * 255, spacing).astype(np.uint8) for sl in sub])
    return (images, real)

def volume_to_triplets(cfg, volume, spacing, plane, laterality, thickness):
    if getattr(cfg, 'normalize_laterality', True) and laterality == 'R':
        volume = volume[::-1] if plane == 'SAG' else [sl[:, ::-1] for sl in volume]
    reference, valid = slice_plan(len(volume), cfg.normalization_reference_slices)
    pixels = np.concatenate([volume[index].astype(np.float32).ravel() for index, real in zip(reference, valid) if real > 0])
    lo, hi = np.percentile(pixels, [cfg.pct_low, cfg.pct_high])
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    lo, span = (np.float32(lo), np.float32(hi - lo))
    centers, real = slice_plan(len(volume), cfg.slices_per_slot)
    step = max(1, int(round(cfg.triplet_delta_mm / max(float(np.float32(thickness)), 0.5))))
    indices = np.clip(centers[:, None] + np.array([-step, 0, step]), 0, len(volume) - 1)
    images = {}
    for index in np.unique(indices):
        image = np.clip((volume[index].astype(np.float32) - lo) / span, 0, 1) * 255.0
        images[index] = physical_crop_resize(cfg, image, spacing).astype(np.uint8)
    triplets = np.stack([images[index] for index in indices.ravel()])
    return (triplets.reshape(cfg.slices_per_slot, 3, cfg.img_px, cfg.img_px), real)

def read_triplet_study(cfg, uid, series, image_root):
    slots = getattr(cfg, 'slot_names', SLOTS)
    selected, scans, laterality = scan_study(uid, series, image_root, len(slots) - len(SLOTS), getattr(cfg, 'use_acquisition_metadata', True))
    images = np.zeros((len(slots), cfg.slices_per_slot, 3, cfg.img_px, cfg.img_px), np.uint8)
    mask = np.zeros(len(slots), np.float32)
    smask = np.zeros((len(slots), cfg.slices_per_slot), np.float32)
    for row in selected.itertuples(index=False):
        index = getattr(row, 'slot_id', SLOTS.index(row.slot))
        scan = scans[row.SeriesInstanceUID]
        try:
            volume = decode_volume(scan['paths'])
            slot_images, slot_mask = volume_to_triplets(cfg, volume, row.spacing_mm, row.plane, laterality, row.z_mm)
        except Exception as exc:
            print(f'Unusable series {uid}/{row.SeriesInstanceUID}: {type(exc).__name__}: {exc}', flush=True)
            continue
        images[index], smask[index] = (slot_images, slot_mask)
        mask[index] = 1
    return (images, mask, smask)

def decode_volume(paths):
    slices, good = ([], [])
    for index, path in enumerate(paths):
        try:
            ds = pydicom.dcmread(path, force=True)
            image = ds.pixel_array.astype(np.float32, copy=False)
            image = image * float(getattr(ds, 'RescaleSlope', 1.0) or 1.0)
            image = image + float(getattr(ds, 'RescaleIntercept', 0.0) or 0.0)
            if str(getattr(ds, 'PhotometricInterpretation', '')).upper() == 'MONOCHROME1':
                image = float(image.max()) - image
            if image.ndim != 2 or not image.size or (not np.isfinite(image).all()):
                raise ValueError('Invalid decoded slice')
        except Exception as exc:
            print(f'Unreadable slice {path}: {type(exc).__name__}: {exc}', flush=True)
            image = None
        slices.append(image)
        if image is not None:
            good.append(index)
    if not good:
        raise ValueError('No decodable slices')
    for index, image in enumerate(slices):
        if image is None:
            nearest = good[int(np.argmin(np.abs(np.asarray(good) - index)))]
            slices[index] = slices[nearest]
    value_min, value_max = (min((float(sl.min()) for sl in slices)), max((float(sl.max()) for sl in slices)))
    if value_max <= value_min:
        return [np.zeros(sl.shape, dtype=np.uint16) for sl in slices]
    scale = (value_max - value_min) / 65535.0
    for index, image in enumerate(slices):
        slices[index] = np.rint((image - value_min) / scale).clip(0, 65535).astype(np.uint16)
    return slices

def read_study(cfg, uid, series, image_root):
    if cfg.dataset == 'ds_knee_cached_triplet':
        from copy import copy
        from ds_knee_cached_triplet import cached_triplets
        shared = copy(cfg)
        shared.dataset = 'ds_knee_extra'
        images, mask, smask = read_study(shared, uid, series, image_root)
        images, smask = cached_triplets(images, smask, cfg.slices_per_slot, cfg.slice_offsets)
        return (images, mask, smask)
    if cfg.dataset == 'ds_knee_triplet':
        return read_triplet_study(cfg, uid, series, image_root)
    slots = getattr(cfg, 'slot_names', SLOTS) if cfg.dataset == 'ds_knee_extra' else SLOTS
    selected, scans, laterality = scan_study(uid, series, image_root, len(slots) - len(SLOTS), getattr(cfg, 'use_acquisition_metadata', True))
    image_px = getattr(cfg, 'cache_img_px', cfg.img_px)
    images = np.zeros((len(slots), cfg.cache_slices, 1, image_px, image_px), dtype=np.uint8)
    mask = np.zeros(len(slots), dtype=np.float32)
    smask = np.zeros((len(slots), cfg.cache_slices), dtype=np.float32)
    for row in selected.itertuples(index=False):
        scan = scans[row.SeriesInstanceUID]
        index = getattr(row, 'slot_id', SLOTS.index(row.slot))
        try:
            volume = decode_volume(scan['paths'])
            slot_images, slot_mask = volume_to_slot(cfg, volume, row.spacing_mm, row.slot, laterality)
        except Exception as exc:
            print(f'Unusable series {uid}/{row.SeriesInstanceUID}: {type(exc).__name__}: {exc}', flush=True)
            continue
        images[index, :, 0], smask[index] = (slot_images, slot_mask)
        mask[index] = 1
    images = crop_study_images(cfg, images, mask)
    return (images, mask, smask)

def image_digest(images, mask, smask):
    digest = hashlib.sha256()
    for array in (images, mask, smask):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()

class DicomDataset(Dataset):

    def __init__(self, uids, cfg, competition_root, split):
        self.uids = list(uids)
        self.cfg = cfg
        root = Path(competition_root)
        series = pd.read_csv(root / f'{split}_series.csv')
        self.empty_series = series.iloc[:0]
        self.series = {uid: frame for uid, frame in series.loc[series.StudyInstanceUID.isin(self.uids)].groupby('StudyInstanceUID')}
        self.image_root = root / f'{split}_series'

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, index):
        images, mask, smask = read_study(self.cfg, self.uids[index], self.series.get(self.uids[index], self.empty_series), self.image_root)
        return {'images': torch.from_numpy(images), 'mask': torch.from_numpy(mask), 'smask': torch.from_numpy(smask), 'uid': self.uids[index], 'digest': image_digest(images, mask, smask)}

class CachedDataset(Dataset):

    def __init__(self, uids, cfg):
        self.cfg = cfg
        self.uids = list(uids)
        if cfg.dataset in ('ds_knee_extra', 'ds_knee_cached_triplet'):
            if cfg.dataset == 'ds_knee_cached_triplet':
                from ds_knee_cached_triplet import CustomDataset
            else:
                from ds_knee_extra import CustomDataset
            frame = pd.read_csv(cfg.train_df).set_index('StudyInstanceUID').loc[self.uids].reset_index()
            self.extra = CustomDataset(frame, cfg, mode='val')
            return
        if cfg.dataset == 'ds_knee_triplet':
            from ds_knee_triplet import CustomDataset, read_cache_manifest
            read_cache_manifest(cfg)
            frame = pd.read_csv(cfg.train_df).set_index('StudyInstanceUID').loc[self.uids].reset_index()
            self.triplets = CustomDataset(frame, cfg, mode='val')
            return
        root = Path(cfg.cache_dir)
        self.images = np.load(root / 'images.npy', mmap_mode='r')
        self.mask = np.load(root / 'mask.npy', mmap_mode='r')
        self.smask = np.load(root / 'smask.npy', mmap_mode='r')
        frame = pd.read_csv(root / 'studies.csv')
        lookup = dict(zip(frame.StudyInstanceUID, range(len(frame))))
        self.rows = [lookup[uid] for uid in uids]

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, index):
        if self.cfg.dataset in ('ds_knee_extra', 'ds_knee_cached_triplet'):
            item = self.extra[index]
            images, mask, smask = (item[key].numpy() for key in ('images', 'mask', 'smask'))
            return {'images': item['images'], 'mask': item['mask'], 'smask': item['smask'], 'uid': self.uids[index], 'digest': image_digest(images, mask, smask)}
        if self.cfg.dataset == 'ds_knee_triplet':
            item = self.triplets[index]
            images, mask, smask = (item[key].numpy() for key in ('images', 'mask', 'smask'))
            return {'images': item['images'], 'mask': item['mask'], 'smask': item['smask'], 'uid': self.uids[index], 'digest': image_digest(images, mask, smask)}
        row = self.rows[index]
        images = self.images[row][:, :, None].copy()
        mask, smask = (self.mask[row].copy(), self.smask[row].copy())
        images = crop_study_images(self.cfg, images, mask)
        return {'images': torch.from_numpy(images), 'mask': torch.from_numpy(mask), 'smask': torch.from_numpy(smask), 'uid': self.uids[index], 'digest': image_digest(images, mask, smask)}
