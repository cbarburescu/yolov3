{
    'hyp': {'giou': 3.69,  # giou loss gain
            'cls': 39.8,  # cls loss gain
            'cls_pw': 0.998,  # cls BCELoss positive_weight
            'obj': 63.9,  # obj loss gain (*=img_size/320 if img_size != 320)
            'obj_pw': 0.984,  # obj BCELoss positive_weight
            'iou_t': 0.183,  # iou training threshold
            'lr0': 9.83e-4,  # initial learning rate (SGD=5E-3, Adam=5E-4)
            'lrf': 5e-6,  # final learning rate (with cos scheduler)
            'momentum': 0.898,  # SGD momentum
            'nesterov': True,
            'weight_decay': 4.57e-4,  # optimizer weight decay
            # focal loss gamma (efficientDet default is gamma=1.5)
            'fl_gamma': 0.0,
            'hsv_h': 0.0137,  # image HSV-Hue augmentation (fraction)
            'hsv_s': 0.672,  # image HSV-Saturation augmentation (fraction)
            'hsv_v': 0.335,  # image HSV-Value augmentation (fraction)
            'degrees': 10,  # image rotation (+/- deg)
            'translate': 0.1,  # image translation (+/- fraction)
            'scale': 0.1,  # image scale (+/- gain)
            'shear': 3},  # image shear (+/- deg)
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
        "layers": {"quant": ["conv", "activation"]},
    }
}
