import types
import torch
import importlib
from .compression import topk_compress
import torch.distributed as dist

class ComFusedAdam(torch.optim.Optimizer):

    """Implements Adam algorithm. Currently GPU-only.  Requires Apex to be installed via
    ``python setup.py install --cuda_ext --cpp_ext``.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params,
                 lr=1e-3, bias_correction = True,
                 betas=(0.9, 0.999), eps=1e-8, eps_inside_sqrt = False,
                 weight_decay=0., max_grad_norm=0., amsgrad=False):
        global com_fused_adam_cuda
        com_fused_adam_cuda = importlib.import_module("com_fused_adam_cuda")
        global fused_adam_cuda
        fused_adam_cuda = importlib.import_module("fused_adam_cuda")
        print("Using compressed update")

        if amsgrad:
            raise RuntimeError('FusedAdam does not support the AMSGrad variant.')
        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(ComFusedAdam, self).__init__(params, defaults)
        self.eps_mode = 0 if  eps_inside_sqrt else 1
        self.global_iter = 0

    def step(self, closure=None, grads=None, output_params=None, scale=1., grad_norms=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
            output params (list of tensors, optional): A reduced precision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients. (default: None)
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()

        if grads is None:
            grads_group = [None]*len(self.param_groups)
        # backward compatibility
        # assuming a list/generator of parameter means single group
        elif isinstance(grads, types.GeneratorType):
            grads_group = [grads]
        elif type(grads[0])!=list:
            grads_group = [grads]
        else:
            grads_group = grads

        if output_params is None:
            output_params_group = [None]*len(self.param_groups)
        elif isinstance(output_params, types.GeneratorType):
            output_params_group = [output_params]
        elif type(output_params[0])!=list:
            output_params_group = [output_params]
        else:
            output_params_group = output_params

        if grad_norms is None:
            grad_norms = [None]*len(self.param_groups)

        for group, grads_this_group, output_params_this_group, grad_norm_group in zip(self.param_groups, grads_group, output_params_group, grad_norms):
            if grads_this_group is None:
               grads_this_group = [None]*len(group['params'])
            if output_params_this_group is None:
               output_params_this_group = [None]*len(group['params'])

            if grad_norm_group is None:
                grad_norm_group = [None] * len(group['params'])
            elif not isinstance(grad_norm_group, list):
                grad_norm_group = [grad_norm_group]

            bias_correction = 1 if group['bias_correction'] else 0

            for p, grad, output_param, grad_norm in zip(group['params'], grads_this_group, output_params_this_group, grad_norm_group):
                
                # compute combined scale factor for this group
                combined_scale = scale
                if group['max_grad_norm'] > 0:
                    # norm is in fact norm*scale
                    clip = ((grad_norm / scale) + 1e-6) / group['max_grad_norm']
                    if clip > 1:
                        combined_scale = clip * scale
                
                #print("Combined Scale is ", combined_scale)

                #note: p.grad should not ever be set for correct operation of mixed precision optimizer that sometimes sends None gradients
                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('FusedAdam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Initialized the error buffer
                    state['model_update'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                model_update = state['model_update']
                
                if self.global_iter >60000000:
                    model_update = state['model_update']
                    fullgrad = grad.float()/combined_scale
                    #scaled_error = model_update*combined_scale
                    temp_grad = topk_compress(fullgrad + model_update  ,alpha=0.03)
                    state['model_update'] = fullgrad + model_update - temp_grad
                    #grad = temp_grad
                    dist.all_reduce(temp_grad/dist.get_world_size())
                    print("Using compressed gradients")
                    grad = (temp_grad*combined_scale).half()
                   # print('OK')
                   #test_mask = grad == 0.0
                   # print('After:  ', test_mask.sum()) 
                else:
                   # dist.all_reduce(grad)
                   # grad=grad/dist.get_world_size()
                    pass

                state['step'] += 1

                out_p = torch.tensor([], dtype = torch.float) if output_param is None else output_param
                fused_adam_cuda.adam(p.data,
                                     out_p,
                                     exp_avg,
                                     exp_avg_sq,
                                     grad,
                                     group['lr'],
                                     beta1,
                                     beta2,
                                     group['eps'],
                                     combined_scale,
                                     state['step'],
                                     self.eps_mode,
                                     bias_correction,
                                     group['weight_decay'])
        self.global_iter +=1
        return loss