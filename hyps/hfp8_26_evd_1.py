# 1.98       38.3      0.936       95.7       1.09      0.264   0.000615      5e-06      0.844          1   0.000659          0     0.0121      0.277       0.27          0          0          0          0
{
    'hyp': {'giou': 1.98,  # giou loss gain
            'cls': 38.3,  # cls loss gain
            'cls_pw': 0.936,  # cls BCELoss positive_weight
            'obj': 95.7,  # obj loss gain (*=img_size/320 if img_size != 320)
            'obj_pw': 1.09,  # obj BCELoss positive_weight
            'iou_t': 0.264,  # iou training threshold
            'lr0': 6.15e-4,  # initial learning rate (SGD=5E-3, Adam=5E-4)
            'lrf': 5e-6,  # final learning rate (with cos scheduler)
            'momentum': 0.844,  # SGD momentum
            'nesterov': True,
            'weight_decay': 6.59e-4,  # optimizer weight decay
            # focal loss gamma (efficientDet default is gamma=1.5)
            'fl_gamma': 0.0,
            'hsv_h': 0.021,  # image HSV-Hue augmentation (fraction)
            'hsv_s': 0.277,  # image HSV-Saturation augmentation (fraction)
            'hsv_v': 0.270,  # image HSV-Value augmentation (fraction)
            'degrees': 1.98 * 0,  # image rotation (+/- deg)
            'translate': 0.05 * 0,  # image translation (+/- fraction)
            'scale': 0.05 * 0,  # image scale (+/- gain)
            'shear': 0.641 * 0},  # image shear (+/- deg)
    'quant_hyp': {
        # forward pass quantization
        "forward": {"exp": 4, "man": 3, "bias": 4, "rounding": "nearest"},
        # backward pass quantization
        "backward": {"exp": 5, "man": 2, "rounding": "nearest"},
        # weight quantization (after optim.step())
        "weight": {"exp": 4, "man": 3, "bias": 4, "rounding": "nearest"},
        # gradient quantization (after loss.backwards(), before optim.step())
        "grad": {"exp": 5, "man": 2, "rounding": "nearest", "scale": "dynamic"},
        # momentum quantization (after optim.step())
        "mom": {"exp": 6, "man": 9, "rounding": "stochastic"},
        # accumulator quantization (after optim.step())
        "acc": {"exp": 6, "man": 9, "rounding": "stochastic"},
        # loss scaling (before loss.backwards())
        "loss": {"scale": "amp_dynamic"},
        # type of layers to insert a quantization layer after and exceptions
        "layers": {"quant": ["conv", "normalization", "activation"]},
    }
}
