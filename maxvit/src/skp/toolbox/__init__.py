"""Optional utilities for analysis, preprocessing, and inference.

Nothing in the core training path should depend on this package. Toolbox modules
are intentionally importable à la carte so optional dependencies such as pydicom
or dicomsdl are only required when their functions are used.
"""
