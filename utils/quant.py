from apex import amp
from qtorch import FloatingPoint
from qtorch.auto_low import lower, sequential_lower
from qtorch.optim import OptimLP
from qtorch.quant import float_quantize


def quant(quant_hyp, model, optimizer=None, amp=False):
    # Model
    forward_rounding = quant_hyp["forward"].pop("rounding")
    backward_rounding = quant_hyp["backward"].pop("rounding")
    forward_num = FloatingPoint(**quant_hyp["forward"])
    backward_num = FloatingPoint(**quant_hyp["backward"])
    except_layers = quant_hyp["layers"].get("except", [])
    model = sequential_lower(model,
                            layer_types=quant_hyp["layers"]["quant"],
                            except_layers=except_layers,
                            forward_number=forward_num,
                            forward_rounding=forward_rounding,
                            backward_number=backward_num,
                            backward_rounding=backward_rounding,
                            )

    # Optimizer
    if optimizer is not None:
        grad_scaling = quant_hyp["grad"].pop("scale")
        if isinstance(grad_scaling, str):
            grad_scaling = 1

        # Weight and grad
        weight_quant = lambda x: float_quantize(x, **quant_hyp["weight"])
        gradient_quant = lambda x: float_quantize(x, **quant_hyp["grad"])

        # Acc and mom (optional)
        acc_quant = (lambda x: float_quantize(x, **quant_hyp["acc"])) if quant_hyp.get("acc") is not None else None
        mom_quant = (lambda x: float_quantize(x, **quant_hyp["mom"])) if quant_hyp.get("mom") is not None else None

        optimizer = optim.SGD(
            model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=hyp['nesterov'])
        optimizer = OptimLP(optimizer,
                            weight_quant=weight_quant,
                            grad_quant=gradient_quant,
                            grad_scaling=grad_scaling,
                            momentum_quant=mom_quant,
                            acc_quant=acc_quant)
    if amp:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O0', loss_scale="dynamic")

    if optimizer is None:
        return model
    else:
        return model, optimizer