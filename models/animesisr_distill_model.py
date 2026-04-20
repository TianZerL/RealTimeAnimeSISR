import torch

from collections import OrderedDict

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils.registry import MODEL_REGISTRY

from .animesisr_net_model import AnimeSISRNetModel


@MODEL_REGISTRY.register()
class AnimeSISRDistillModel(AnimeSISRNetModel):
    def __init__(self, opt):
        super().__init__(opt)

        # define network
        self.net_t = build_network(opt['network_t'])
        self.net_t = self.model_to_device(self.net_t)
        self.print_network(self.net_t)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_t', None)
        if load_path is None:
            raise ValueError('pretrain_network_t is None.')

        param_key = self.opt['path'].get('param_key_t', 'params')
        self.load_network(self.net_t, load_path, self.opt['path'].get('strict_load_t', True), param_key)

        self.net_t.eval().requires_grad_(False)

    def init_training_settings(self):
        train_opt = self.opt['train']

        if cri_type := train_opt.get('distill_opt'):
            self.cri_distill = build_loss(cri_type).to(self.device)
        else:
            raise ValueError('distill losses is None.')

        super().init_training_settings()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        with torch.no_grad():
            self.target = self.net_t(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        l_distill = self.cri_distill(self.output, self.target)
        l_total += l_distill
        loss_dict['l_distill'] = l_distill

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # edge loss
        if self.cri_edge:
            l_edge = self.cri_edge(self.output, self.gt)
            l_total += l_edge
            loss_dict['l_edge'] = l_edge
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
