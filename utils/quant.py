from apex import amp
from copy import deepcopy
from qtorch import FloatingPoint
from qtorch.auto_low import lower, sequential_lower
from qtorch.optim import OptimLP
from qtorch.quant import float_quantize
import torch.optim as optim


def quant(quant_hyp, model, hyp=None, optimizer=None, use_amp=False):
    # Model
    quant_hyp_copy = deepcopy(quant_hyp)

    forward_rounding = quant_hyp_copy["forward"].pop("rounding")
    backward_rounding = quant_hyp_copy["backward"].pop("rounding")
    forward_num = FloatingPoint(**quant_hyp_copy["forward"])
    backward_num = FloatingPoint(**quant_hyp_copy["backward"])
    except_layers = quant_hyp_copy["layers"].get("except", [])
    model = sequential_lower(model,
                            layer_types=quant_hyp_copy["layers"]["quant"],
                            except_layers=except_layers,
                            forward_number=forward_num,
                            forward_rounding=forward_rounding,
                            backward_number=backward_num,
                            backward_rounding=backward_rounding,
                            )

    # Optimizer
    if optimizer is not None:
        grad_scaling = quant_hyp_copy["grad"].pop("scale")
        if isinstance(grad_scaling, str):
            grad_scaling = 1

        # Weight and grad
        weight_quant = lambda x: float_quantize(x, **quant_hyp_copy["weight"])
        gradient_quant = lambda x: float_quantize(x, **quant_hyp_copy["grad"])

        # Acc and mom (optional)
        acc_quant = (lambda x: float_quantize(x, **quant_hyp_copy["acc"])) if quant_hyp_copy.get("acc") is not None else None
        mom_quant = (lambda x: float_quantize(x, **quant_hyp_copy["mom"])) if quant_hyp_copy.get("mom") is not None else None

        optimizer = optim.SGD(
            model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=hyp['nesterov'])
        optimizer = OptimLP(optimizer,
                            weight_quant=weight_quant,
                            grad_quant=gradient_quant,
                            grad_scaling=grad_scaling,
                            momentum_quant=mom_quant,
                            acc_quant=acc_quant)
    if use_amp:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O0', loss_scale="dynamic")

    if optimizer is None:
        return model
    else:
        return model, optimizer