from segmentation_ShapeNetPart.hycore_seg_v5.losses import HyCoReSegLoss as _Loss

class HyCoReSegLoss(_Loss):
    """Zero active losses; detached geometry and independent samplers retained."""
    def __init__(self, alpha=0., beta=0., *args, **kwargs):
        if (alpha,beta)!=(0.,0.):
            raise ValueError('F0/G0/H0 require alpha=beta=0')
        super().__init__(alpha,beta,*args,**kwargs)
