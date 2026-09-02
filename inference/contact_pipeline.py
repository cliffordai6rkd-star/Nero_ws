"""Compatibility exports for the canonical Contact World Model pipeline.

WM inference is implemented once in :mod:`inference.contact_wm_pipeline`.
This module remains importable for applications that used the historical
``inference.contact_pipeline`` path.
"""

from inference.contact_wm_pipeline import (
    ContactWMInferencePipeline,
    ContactWorldModelInferencePipeline,
)

ContactInferencePipeline = ContactWMInferencePipeline

__all__ = [
    "ContactWMInferencePipeline",
    "ContactWorldModelInferencePipeline",
    "ContactInferencePipeline",
]
