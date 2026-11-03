import torch.nn as nn


class ParameterGrouper:
    def __init__(self, groups: list):
        """
        Logic for splitting model parameters into different optimizer groups based on their names.

        Args:
            groups (list[dict]): A list of dictionaries defining the grouping rules.
                Each dict can contain:
                - 'match' (str): The substring to look for in the parameter name (e.g., "encoder").
                - 'lr_scale' (float): Multiplier for the global base_lr.
                - 'lr' (float): Absolute learning rate (overrides lr_scale).
                - 'weight_decay' (float): Specific weight decay (optional).

        Example:
        grouper = ParameterGrouper([
            {
                'match': 'fc', # Only the final fully connected layer
                'lr_scale': 1.0
            },
            {
                'match': 'layer4', # Top convolutional block
                'lr_scale': 0.1
            }
            # Everything else (layer1, layer2...) falls into 'default'
            # You can set base_lr very low, or use this grouper to filter them out.
        ])

        IMPORTANT - MATCHING LOGIC (ORDER MATTERS):
        -------------------------------------------
        The matching is performed sequentially (First Match Wins).
        Once a parameter matches a group's substring, it is assigned to that group
        and the code stops checking subsequent groups.

        You must list MORE SPECIFIC rules BEFORE general rules.

        Example:
            If you want the 'encoder' to have LR=1e-5, but 'encoder.layer4' to have LR=1e-3:

            CORRECT ORDER:
            [
                {'match': 'layer4',  'lr': 1e-3},  # Specific override first
                {'match': 'encoder', 'lr': 1e-5}   # General rule second
            ]

            INCORRECT ORDER:
            [
                {'match': 'encoder', 'lr': 1e-5},  # Catches everything in encoder (including layer4)
                {'match': 'layer4',  'lr': 1e-3}   # Never reached
            ]
        """
        self.groups = groups

    def __call__(self, model: nn.Module, base_lr: float, base_wd: float):
        """
        Categorizes the model's parameters and returns the list of groups expected by PyTorch optimizers.
        """
        # 1. Bucket parameters
        categorized_params = {i: [] for i in range(len(self.groups))}
        default_params = []

        # 2. Iterate over all named parameters
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Find which group this parameter belongs to
            assigned = False
            for i, group_cfg in enumerate(self.groups):
                # Check if substring is in the full parameter name
                if group_cfg["match"] in name:
                    categorized_params[i].append(param)
                    assigned = True
                    break  # Stop checking after first match (Priority Logic)

            if not assigned:
                default_params.append(param)

        # 3. Construct the optimizer input list
        optimizer_groups = []

        # Add specific groups
        for i, params in categorized_params.items():
            if not params:
                continue

            cfg = self.groups[i]

            # Determine LR: Absolute or Scaled
            if "lr" in cfg:
                final_lr = cfg["lr"]
            else:
                final_lr = base_lr * cfg.get("lr_scale", 1.0)

            # Determine WD: Override or Base
            final_wd = cfg.get("weight_decay", base_wd)

            optimizer_groups.append(
                {
                    "params": params,
                    "lr": final_lr,
                    "weight_decay": final_wd,
                    "name": cfg["match"],  # Helpful for debugging
                }
            )

        # Add default group (leftovers)
        if default_params:
            optimizer_groups.append(
                {
                    "params": default_params,
                    "lr": base_lr,
                    "weight_decay": base_wd,
                    "name": "default",
                }
            )

        return optimizer_groups
