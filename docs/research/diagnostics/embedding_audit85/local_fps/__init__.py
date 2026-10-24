"""Evaluation-only access to the repository's original CUDA FPS kernel."""

from .fps import OriginalFPS, install_pointnet2_fps

__all__ = ["OriginalFPS", "install_pointnet2_fps"]
