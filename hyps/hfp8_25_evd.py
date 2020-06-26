# 3.61       39.1       1.01       65.7      0.983      0.191   0.000994      5e-06      0.898          1   0.000454          0     0.0134      0.672      0.335          0          0          0          0

{
    'hyp': {'giou': 3.61,  # giou loss gain
            'cls': 39.1,  # cls loss gain
            'cls_pw': 1.01,  # cls BCELoss positive_weight
            'obj': 65.7,  # obj loss gain (*=img_size/320 if img_size != 320)
            'obj_pw': 0.983,  # obj BCELoss positive_weight
            'iou_t': 0.191,  # iou training threshold
            'lr0': 9.94e-4,  # initial learning rate (SGD=5E-3, Adam=5E-4)
            'lrf': 5e-6,  # final learning rate (with cos scheduler)
            'momentum': 0.898,  # SGD momentum
            'nesterov': True,
            'weight_decay': 0.000454,  # optimizer weight decay
            # focal loss gamma (efficientDet default is gamma=1.5)
            'fl_gamma': 0.0,
            'hsv_h': 0.0134,  # image HSV-Hue augmentation (fraction)
            'hsv_s': 0.672,  # image HSV-Saturation augmentation (fraction)
            'hsv_v': 0.335,  # image HSV-Value augmentation (fraction)
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
        "layers": {"quant": ["conv", "activation"]},
    }
}
