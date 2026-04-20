import torch

from collections import OrderedDict

from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class AnimeSISRNetModel(SRModel):
    def init_training_settings(self):
        train_opt = self.opt['train']

        if cri_type := train_opt.get('edge_opt'):
            self.cri_edge = build_loss(cri_type).to(self.device)
        else:
            self.cri_edge = None

        super().init_training_settings()

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'en' in data:
            self.gt = data['en'].to(self.device)
        else:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
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

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                if hasattr(self.net_g_ema, 'reparameterize'):
                    self.output = self.net_g_ema.reparameterize().eval()(self.lq)
                else:
                    self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                if hasattr(self.net_g, 'reparameterize'):
                    self.output = self.net_g.reparameterize().eval()(self.lq)
                else:
                    self.output = self.net_g(self.lq)
            self.net_g.train()
