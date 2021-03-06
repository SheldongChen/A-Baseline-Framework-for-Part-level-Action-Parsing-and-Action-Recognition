from torch.nn import functional as F


def upsample_like(source, target, mode='bilinear', align_corners=False):
    """Upsample the source to the shape of the target.

    Upsample the source to the shape of target. The input must be a Tensor,
    but the target can be a Tensor or a np.ndarray with the shape
    (..., target_h, target_w).

    Args:
        source (Tensor): A 3D/4D Tensor with the shape (N, H, W) or
            (N, C, H, W).
        target (Tensor | np.ndarray): The upsampling target with the shape
            (..., target_h, target_w).
        mode (str): Algorithm used for upsampling. The options are the same
            as those in F.interpolate(). Default: ``'bilinear'``.
        align_corners (bool): The same as the argument in F.interpolate().

    Returns:
        Tensor: The upsampled source Tensor.
    """
    assert len(target.shape) >= 2

    def _upsample_like(source, target, mode='bilinear', align_corners=False):
        """Upsample the source (4D) to the shape of the target."""
        target_h, target_w = target.shape[-2:]
        source_h, source_w = source.shape[-2:]
        if target_h != source_h or target_w != source_w:
            source = F.interpolate(
                source,
                size=(target_h, target_w),
                mode=mode,
                align_corners=align_corners)
        return source

    if len(source.shape) == 3:
        source = source[:, None, :, :]
        source = _upsample_like(source, target, mode, align_corners)
        return source[:, 0, :, :]
    else:
        return _upsample_like(source, target, mode, align_corners)
