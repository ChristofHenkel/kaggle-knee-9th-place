from pathlib import Path

import numpy as np


def calc_metric(cfg, pp_out, val_df, pre='val'):
    probabilities = pp_out['predictions'].detach().cpu().numpy()
    output = Path(cfg.output_dir)
    val_df.to_csv(output / 'series_predictions_index.csv', index=False)
    np.savez_compressed(output / 'moaks_predictions.npz', probabilities=probabilities,
                        StudyInstanceUID=val_df.StudyInstanceUID.to_numpy(str),
                        SeriesInstanceUID=val_df.SeriesInstanceUID.to_numpy(str))
    return {}
