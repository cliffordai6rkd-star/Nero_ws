"""Backward-compatible imports for the canonical Contact WM pipeline.

Use :mod:`inference.contact_wm_pipeline` for new code.  No SWM or torque
world-model implementation is retained here.
"""

from inference.contact_wm_pipeline import (
    ContactWMInferencePipeline,
    ContactWorldModelInferencePipeline,
    ContactWMPipeline,
)

SWMInferencePipeline = ContactWMInferencePipeline
SWMPipeline = ContactWMPipeline
TorqueWorldModelInferencePipeline = ContactWorldModelInferencePipeline

__all__ = [
    "ContactWMInferencePipeline",
    "ContactWorldModelInferencePipeline",
    "ContactWMPipeline",
    "SWMInferencePipeline",
    "SWMPipeline",
    "TorqueWorldModelInferencePipeline",
]
