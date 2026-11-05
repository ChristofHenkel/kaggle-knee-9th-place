"""Validation metrics for RSNA knee models with anchored expert validation."""

from __future__ import annotations

import re
from typing import Any

import torch
from sklearn.metrics import roc_auc_score
from torchmetrics import Metric

from skp.metrics import utils


COMPETITION_LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def _cat_state(state: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    return state if isinstance(state, torch.Tensor) else torch.cat(state, dim=0)


def _metric_name(label: str) -> str:
    value = label.casefold().replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def _auc(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    if target.numel() == 0 or target.unique().numel() < 2:
        return torch.tensor(float("nan"))
    return torch.tensor(
        roc_auc_score(target.numpy(), prediction.numpy()), dtype=torch.float32
    )


class ValidationMetrics(Metric):
    """Compute per-study expert and combined metrics from a series stream.

    Required batch keys are ``y`` (B,C), ``y_mask`` (B,C),
    ``uncertain_mask`` (B,C), ``is_expert`` (B,), and ``study_index`` (B,).
    ``fluid_sensitive`` (B,) is additionally required when configured to
    report fluid-sensitive or non-fluid-sensitive validation views.
    Report-derived ``not_mentioned`` is 0 and ``uncertain`` is the soft target
    0.5 for loss. AUC evaluates uncertainty under both binary mappings.
    """

    def __init__(self, cfg: Any, dist_sync_on_step: bool = False):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            sync_on_compute=False,
        )
        class_names = cfg.require("class_names")
        if not isinstance(class_names, list) or not all(
            isinstance(name, str) and name for name in class_names
        ):
            raise ValueError("cfg.class_names must be a non-empty list of strings")
        if len(class_names) != len(set(class_names)):
            raise ValueError("cfg.class_names contains duplicates")
        missing = [label for label in COMPETITION_LABELS if label not in class_names]
        if missing:
            raise ValueError(f"Missing competition labels: {missing}")
        self.class_names = class_names
        self.report_thresholds = cfg.get("validation_report_thresholds")
        if self.report_thresholds is not None and (
            not isinstance(self.report_thresholds, list)
            or not self.report_thresholds
            or any(not isinstance(t, (int, float)) or isinstance(t, bool)
                   or not 0 < t < 1 for t in self.report_thresholds)
            or len(set(self.report_thresholds)) != len(self.report_thresholds)
        ):
            raise ValueError("validation_report_thresholds must be unique thresholds in (0,1)")
        self.evaluation_labels = list(
            cfg.get("validation_target_labels") or COMPETITION_LABELS
        )
        if (
            not self.evaluation_labels
            or len(set(self.evaluation_labels)) != len(self.evaluation_labels)
            or any(label not in COMPETITION_LABELS for label in self.evaluation_labels)
        ):
            raise ValueError(
                "validation_target_labels must be a unique competition subset"
            )
        self.macro_name = (
            "competition_macro_auc"
            if self.evaluation_labels == COMPETITION_LABELS
            else "specialist_macro_auc"
        )
        self.competition_indices = [
            class_names.index(label) for label in self.evaluation_labels
        ]
        self.auxiliary_indices = [
            index
            for index, label in enumerate(class_names)
            if label not in COMPETITION_LABELS
        ]
        selection_labels = cfg.get("expert_selection_labels")
        if selection_labels is None:
            selection_labels = list(self.evaluation_labels)
        if (
            not isinstance(selection_labels, list)
            or not selection_labels
            or len(selection_labels) != len(set(selection_labels))
            or any(label not in self.evaluation_labels for label in selection_labels)
        ):
            raise ValueError(
                "cfg.expert_selection_labels must be a unique non-empty subset "
                "of competition labels"
            )
        self.expert_selection_labels = selection_labels
        self.series_views = cfg.get("validation_series_views", None) or ["all_series"]
        allowed_views = {"all_series", "fluid_sensitive", "non_fluid_sensitive"}
        if (
            not isinstance(self.series_views, list)
            or not self.series_views
            or len(self.series_views) != len(set(self.series_views))
            or any(view not in allowed_views for view in self.series_views)
        ):
            raise ValueError(
                "cfg.validation_series_views must be a unique non-empty list drawn "
                f"from {sorted(allowed_views)}"
            )
        self.prefix_series_views = len(self.series_views) > 1
        plane_names = {"Axial": 0, "Coronal": 1, "Sagittal": 2}
        plane_filters = cfg.get("validation_label_plane_filters") or {}
        if not isinstance(plane_filters, dict) or any(
            label not in class_names
            or not isinstance(planes, list)
            or not planes
            or len(planes) != len(set(planes))
            or any(plane not in plane_names for plane in planes)
            for label, planes in plane_filters.items()
        ):
            raise ValueError(
                "validation_label_plane_filters must map known labels to unique, "
                "non-empty lists drawn from Axial, Coronal, and Sagittal"
            )
        allowed_label_planes = torch.ones(len(class_names), 3, dtype=torch.bool)
        for label, planes in plane_filters.items():
            index = class_names.index(label)
            allowed_label_planes[index] = False
            allowed_label_planes[index, [plane_names[plane] for plane in planes]] = True
        self.register_buffer("allowed_label_planes", allowed_label_planes)
        self.has_label_plane_filters = bool(plane_filters)
        self.report_pf_oa_no_tibiofemoral_oa = bool(
            cfg.get("report_pf_oa_no_tibiofemoral_oa", False)
        )
        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)
        self.add_state("valid", default=[], dist_reduce_fx=None)
        self.add_state("uncertain", default=[], dist_reduce_fx=None)
        self.add_state("expert", default=[], dist_reduce_fx=None)
        self.add_state("study", default=[], dist_reduce_fx=None)
        self.add_state("fluid_sensitive", default=[], dist_reduce_fx=None)
        self.add_state("anatomical_plane", default=[], dist_reduce_fx=None)
        # Retain validation identity and the already-computed auxiliary head so
        # held-out predictions can be saved before metric state is reset.
        self.add_state("series_index", default=[], dist_reduce_fx=None)
        self.add_state("aux_p", default=[], dist_reduce_fx=None)

    def update(self, out: dict[str, Any], batch: dict[str, Any]) -> None:
        required = [
            key
            for key in (
                "y",
                "y_mask",
                "uncertain_mask",
                "is_expert",
                "study_index",
            )
            if key not in batch
        ]
        if any(view != "all_series" for view in self.series_views):
            if "fluid_sensitive" not in batch:
                required.append("fluid_sensitive")
        if self.has_label_plane_filters and "anatomical_plane" not in batch:
            required.append("anatomical_plane")
        if required:
            raise KeyError(f"Validation batch missing required keys: {required}")
        logits = out["logits"].detach().float()
        target = batch["y"].detach().float()
        valid = batch["y_mask"].detach().bool()
        uncertain = batch["uncertain_mask"].detach().bool()
        expert = batch["is_expert"].detach().bool().reshape(-1)
        study = batch["study_index"].detach().long().reshape(-1)
        fluid_sensitive = batch.get("fluid_sensitive")
        if fluid_sensitive is None:
            fluid_sensitive = torch.zeros_like(study, dtype=torch.bool)
        else:
            fluid_sensitive = fluid_sensitive.detach().bool().reshape(-1)
        anatomical_plane = batch.get("anatomical_plane")
        if anatomical_plane is None:
            anatomical_plane = torch.full_like(study, -1)
        else:
            anatomical_plane = anatomical_plane.detach().long().reshape(-1)
        series_index = batch.get("index")
        if series_index is None:
            series_index = torch.full_like(study, -1)
        else:
            series_index = series_index.detach().long().reshape(-1)
        auxiliary_logits = out.get("aux_logits")
        if auxiliary_logits is None:
            auxiliary_logits = torch.full_like(logits, float("nan"))
        else:
            auxiliary_logits = auxiliary_logits.detach().float()
        expected_shape = (target.shape[0], len(self.class_names))
        if (
            logits.shape != target.shape
            or target.shape != valid.shape
            or target.shape != uncertain.shape
        ):
            raise ValueError(
                "logits, y, y_mask, and uncertain_mask must share shape; got "
                f"{logits.shape}, {target.shape}, {valid.shape}, {uncertain.shape}"
            )
        if (
            target.shape != expected_shape
            or expert.shape[0] != target.shape[0]
            or study.shape[0] != target.shape[0]
            or fluid_sensitive.shape[0] != target.shape[0]
            or anatomical_plane.shape[0] != target.shape[0]
            or series_index.shape[0] != target.shape[0]
            or auxiliary_logits.shape != target.shape
        ):
            raise ValueError(
                f"Expected target shape {expected_shape} and one study/expert value per row"
            )
        if self.report_thresholds is not None:
            if not torch.isfinite(target).all() or not ((target >= 0) & (target <= 1)).all():
                raise ValueError("Continuous validation targets must be finite in [0,1]")
            expert_targets = target[expert][:, self.competition_indices]
            if not ((expert_targets == 0) | (expert_targets == 1)).all():
                raise ValueError("Expert targets must remain binary")
            if uncertain.any():
                raise ValueError("Continuous validation uses probabilities, not uncertain masks")
        elif not torch.all((target == 0) | (target == 0.5) | (target == 1)):
            raise ValueError("Validation targets must be 0, 0.5, or 1")
        if not valid.all():
            raise ValueError("All targets must be valid; not_mentioned is negative")
        if torch.any(target[uncertain] != 0.5):
            raise ValueError("Uncertain targets must be encoded as 0.5")
        if self.report_thresholds is None and torch.any(target[~uncertain] == 0.5):
            raise ValueError("Only uncertain targets may be encoded as 0.5")
        if self.has_label_plane_filters and torch.any(
            (anatomical_plane < 0) | (anatomical_plane > 2)
        ):
            raise ValueError(
                "Plane-filtered validation requires anatomical_plane indices in [0, 2]"
            )
        self.p.append(logits)
        self.t.append(target)
        self.valid.append(valid)
        self.uncertain.append(uncertain)
        self.expert.append(expert)
        self.study.append(study)
        self.fluid_sensitive.append(fluid_sensitive)
        self.anatomical_plane.append(anatomical_plane)
        self.series_index.append(series_index)
        self.aux_p.append(auxiliary_logits)

    def gather_oof_series(self) -> dict[str, torch.Tensor]:
        """Gather validation-series state on every rank before rank-zero export.

        Call this before ``reset()`` and on *all* distributed ranks: the
        uneven-rank all-gather is collective. Metric calculation remains based
        solely on primary logits; auxiliary logits are diagnostic only.
        """
        states = {
            "logits": self.p,
            "aux_logits": self.aux_p,
            "target": self.t,
            "valid": self.valid,
            "uncertain": self.uncertain,
            "is_expert": self.expert,
            "study_index": self.study,
            "fluid_sensitive": self.fluid_sensitive,
            "anatomical_plane": self.anatomical_plane,
            "series_index": self.series_index,
        }
        return {
            name: utils.distributed_concat(_cat_state(values), dim=0).cpu()
            for name, values in states.items()
        }

    def _key(self, cohort: str, view: str, metric: str) -> str:
        if self.prefix_series_views:
            return f"{cohort}/{view}/{metric}"
        return f"{cohort}/{metric}"

    @staticmethod
    def _aggregate_studies(
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        uncertain: torch.Tensor,
        expert: torch.Tensor,
        study: torch.Tensor,
        anatomical_plane: torch.Tensor | None = None,
        allowed_label_planes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        probabilities = logits.sigmoid()
        grouped_p = []
        grouped_t = []
        grouped_valid = []
        grouped_uncertain = []
        grouped_expert = []
        for study_index in study.unique(sorted=True):
            rows = study == study_index
            first = rows.nonzero(as_tuple=False)[0, 0]
            for name, values in (
                ("target", target),
                ("valid", valid),
                ("uncertain", uncertain),
            ):
                if not torch.all(values[rows] == values[first]):
                    raise RuntimeError(
                        f"Inconsistent {name} within study {study_index}"
                    )
            if not torch.all(expert[rows] == expert[first]):
                raise RuntimeError(
                    f"Inconsistent expert flag within study {study_index}"
                )
            study_probabilities = probabilities[rows]
            if allowed_label_planes is None:
                grouped_p.append(study_probabilities.mean(dim=0))
            else:
                if anatomical_plane is None:
                    raise ValueError(
                        "anatomical_plane is required with allowed_label_planes"
                    )
                study_planes = anatomical_plane[rows]
                eligible = allowed_label_planes[:, study_planes].transpose(0, 1)
                counts = eligible.sum(dim=0)
                filtered = (study_probabilities * eligible).sum(
                    dim=0
                ) / counts.clamp_min(1)
                # A missing preferred plane falls back to the original all-series
                # mean, preserving study coverage without changing prevalence.
                grouped_p.append(
                    torch.where(
                        counts > 0,
                        filtered,
                        study_probabilities.mean(dim=0),
                    )
                )
            grouped_t.append(target[first])
            grouped_valid.append(valid[first])
            grouped_uncertain.append(uncertain[first])
            grouped_expert.append(expert[first])
        return (
            torch.stack(grouped_p),
            torch.stack(grouped_t),
            torch.stack(grouped_valid),
            torch.stack(grouped_uncertain),
            torch.stack(grouped_expert),
        )

    def _compute_view(
        self,
        view: str,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        uncertain: torch.Tensor,
        expert: torch.Tensor,
        study: torch.Tensor,
        fluid_sensitive: torch.Tensor,
        anatomical_plane: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if view == "all_series":
            rows = torch.ones_like(fluid_sensitive, dtype=torch.bool)
        elif view == "fluid_sensitive":
            rows = fluid_sensitive
        else:
            rows = ~fluid_sensitive
        if not rows.any():
            raise RuntimeError(f"Validation series view is empty: {view}")
        prediction, target, valid, uncertain, expert = self._aggregate_studies(
            prediction[rows],
            target[rows],
            valid[rows],
            uncertain[rows],
            expert[rows],
            study[rows],
            anatomical_plane[rows] if anatomical_plane is not None else None,
            self.allowed_label_planes.cpu() if self.has_label_plane_filters else None,
        )
        metrics: dict[str, torch.Tensor] = {}
        competition_index = torch.tensor(self.competition_indices, dtype=torch.long)

        combined = torch.ones_like(expert, dtype=torch.bool)
        for cohort_name, cohort_rows in (("expert", expert), ("combined", combined)):
            metrics[self._key(cohort_name, view, "sample_count")] = (
                cohort_rows.sum().float()
            )
            cohort_valid = valid[cohort_rows][:, competition_index]
            if not cohort_valid.any():
                raise RuntimeError(
                    f"No valid competition targets for {cohort_name}/{view}"
                )
            cohort_prediction = prediction[cohort_rows][:, competition_index]
            cohort_target = target[cohort_rows][:, competition_index]
            loss = torch.nn.functional.binary_cross_entropy(
                cohort_prediction.clamp(1e-7, 1 - 1e-7),
                cohort_target,
                reduction="none",
            )
            metrics[self._key(cohort_name, view, "loss")] = loss[cohort_valid].mean()

        if uncertain[expert][:, competition_index].any():
            raise RuntimeError("Expert competition labels cannot be uncertain")
        if not valid[expert][:, competition_index].all():
            raise RuntimeError("All expert competition labels must be valid")

        expert_aucs = []
        for label, index in zip(
            self.evaluation_labels, self.competition_indices, strict=True
        ):
            auc = _auc(target[expert, index], prediction[expert, index])
            if torch.isnan(auc):
                raise RuntimeError(
                    "Competition AUC is undefined for "
                    f"{self._key('expert', view, label)}"
                )
            metrics[self._key("expert", view, f"auc_{_metric_name(label)}")] = auc
            expert_aucs.append(auc)
        metrics[self._key("expert", view, self.macro_name)] = torch.stack(
            expert_aucs
        ).mean()
        if self.report_pf_oa_no_tibiofemoral_oa:
            pf_index = self.class_names.index("PF OA")
            medial_index = self.class_names.index("Medial OA")
            lateral_index = self.class_names.index("Lateral OA")
            subgroup = (
                expert
                & (target[:, medial_index] == 0)
                & (target[:, lateral_index] == 0)
            )
            subgroup_auc = _auc(
                target[subgroup, pf_index], prediction[subgroup, pf_index]
            )
            if torch.isnan(subgroup_auc):
                raise RuntimeError(
                    "PF OA AUC is undefined in the expert no-tibiofemoral-OA subgroup"
                )
            metrics[self._key("expert", view, "auc_pf_oa_no_tibiofemoral_oa")] = (
                subgroup_auc
            )
            metrics[
                self._key("expert", view, "pf_oa_no_tibiofemoral_oa_sample_count")
            ] = subgroup.sum().float()
            metrics[
                self._key("expert", view, "pf_oa_no_tibiofemoral_oa_positive_count")
            ] = target[subgroup, pf_index].sum()
        selection_aucs = [
            metrics[self._key("expert", view, f"auc_{_metric_name(label)}")]
            for label in self.expert_selection_labels
        ]
        metrics[self._key("expert", view, "selection_macro_auc")] = torch.stack(
            selection_aucs
        ).mean()

        formulations = (
            [(f"threshold_{t:g}", t) for t in self.report_thresholds]
            if self.report_thresholds is not None
            else [("u_negative", 0.0), ("u_positive", 1.0)]
        )
        for uncertainty_name, uncertainty_value in formulations:
            auc_target = torch.where(
                expert[:, None], target, (target >= uncertainty_value).to(target.dtype)
            ) if self.report_thresholds is not None else torch.where(
                uncertain,
                torch.full_like(target, uncertainty_value),
                target,
            )
            competition_aucs = []
            for label, index in zip(
                self.evaluation_labels, self.competition_indices, strict=True
            ):
                auc = _auc(auc_target[:, index], prediction[:, index])
                if torch.isnan(auc):
                    raise RuntimeError(
                        "Competition AUC is undefined for "
                        f"{self._key('combined', view, label)}/{uncertainty_name}"
                    )
                metrics[
                    self._key(
                        "combined",
                        view,
                        f"auc_{_metric_name(label)}_{uncertainty_name}",
                    )
                ] = auc
                competition_aucs.append(auc)
            metrics[
                self._key("combined", view, f"{self.macro_name}_{uncertainty_name}")
            ] = torch.stack(competition_aucs).mean()

            for index in self.auxiliary_indices:
                label = self.class_names[index]
                metrics[
                    self._key(
                        "combined",
                        view,
                        f"auc_aux_{_metric_name(label)}_{uncertainty_name}",
                    )
                ] = _auc(auc_target[:, index], prediction[:, index])
        groups = {
            "medial_compartment": ["MCL", "Medial Meniscus", "Medial OA"],
            "lateral_compartment": ["Lateral Meniscus", "Lateral OA"],
        }
        for group, labels in groups.items():
            if not set(labels).issubset(self.evaluation_labels):
                continue
            for cohort, suffix in [("expert", "")] + [
                ("combined", f"_{name}") for name, _ in formulations
            ]:
                metrics[self._key(cohort, view, f"{group}_auc{suffix}")] = torch.stack(
                    [
                        metrics[
                            self._key(
                                cohort, view, f"auc_{_metric_name(label)}{suffix}"
                            )
                        ]
                        for label in labels
                    ]
                ).mean()
        return metrics

    def compute(self) -> dict[str, torch.Tensor]:
        prediction = utils.distributed_concat(_cat_state(self.p), dim=0).cpu()
        target = utils.distributed_concat(_cat_state(self.t), dim=0).cpu()
        valid = utils.distributed_concat(_cat_state(self.valid), dim=0).cpu().bool()
        uncertain = (
            utils.distributed_concat(_cat_state(self.uncertain), dim=0).cpu().bool()
        )
        expert = utils.distributed_concat(_cat_state(self.expert), dim=0).cpu().bool()
        study = utils.distributed_concat(_cat_state(self.study), dim=0).cpu().long()
        fluid_sensitive = (
            utils.distributed_concat(_cat_state(self.fluid_sensitive), dim=0)
            .cpu()
            .bool()
        )
        anatomical_plane = (
            utils.distributed_concat(_cat_state(self.anatomical_plane), dim=0)
            .cpu()
            .long()
        )
        metrics: dict[str, torch.Tensor] = {}
        for view in self.series_views:
            metrics.update(
                self._compute_view(
                    view,
                    prediction,
                    target,
                    valid,
                    uncertain,
                    expert,
                    study,
                    fluid_sensitive,
                    anatomical_plane,
                )
            )
        return metrics
