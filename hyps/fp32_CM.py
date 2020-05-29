{
    "hyp": {'giou': 3.54,  # giou loss gain
          'cls': 37.4,  # cls loss gain
          'cls_pw': 1.0,  # cls BCELoss positive_weight
          'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
          'obj_pw': 1.0,  # obj BCELoss positive_weight
          'iou_t': 0.20,  # iou training threshold
          'lr0': 1e-3,  # initial learning rate (SGD=5E-3, Adam=5E-4)
          'lrf': 5e-6,  # final learning rate (with cos scheduler)
          'momentum': 0,  # SGD momentum
          'nesterov': False,
          'weight_decay': 0.000484,  # optimizer weight decay
          # focal loss gamma (efficientDet default is gamma=1.5)
          'fl_gamma': 0.0,
          'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
          'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
          'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
          'degrees': 1.98 * 0,  # image rotation (+/- deg)
          'translate': 0.05 * 0,  # image translation (+/- fraction)
          'scale': 0.05 * 0,  # image scale (+/- gain)
          'shear': 0.641 * 0},  # image shear (+/- deg)
    # quant_hyp: {
    #     "forward": {"exp": 5, "man": 2, "rounding": "nearest"},  # forward pass quantization
    #     "backward": {"exp": 5, "man": 2, "rounding": "nearest"},  # backward pass quantization
    #     "weight": {"exp": 5, "man": 2, "rounding": "nearest"},  # weight quantization (after optim.step())
    #     "grad": {"exp": 5, "man": 2, "rounding": "nearest", "scale": 1},  # gradient quantization (after loss.backwards(), before optim.step())
    #     "mom": {"exp": 6, "man": 9, "rounding": "stochastic"},  # momentum quantization (after optim.step())
    #     "acc": {"exp": 6, "man": 9, "rounding": "stochastic"},  # accumulator quantization (after optim.step())
    #     "loss": {"scale": 1000}  # loss scaling (before loss.backwards())
    # }
}
