import torch

from collections import OrderedDict

from basicsr.losses import build_loss
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class AnimeSISRGANModel(SRGANModel):
    def init_training_settings(self):
        train_opt = self.opt['train']

        if cri_type := train_opt.get('edge_opt'):
            self.cri_edge = build_loss(cri_type).to(self.device)
        else:
            self.cri_edge = None

        super().init_training_settings()

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        if 'en' in data:
            self.en = data['en'].to(self.device)
        else:
            self.en = self.gt

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']

        pixel_gt = getattr(self, train_opt.get('pixel_gt', self.gt))
        edge_gt = getattr(self, train_opt.get('edge_gt', self.gt))
        perceptual_gt = getattr(self, train_opt.get('perceptual_gt', self.gt))
        gan_gt = getattr(self, train_opt.get('gan_gt', self.gt))

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, pixel_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # edge loss
            if self.cri_edge:
                l_edge = self.cri_edge(self.output, edge_gt)
                l_total += l_edge
                loss_dict['l_edge'] = l_edge
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, perceptual_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        l_d_fake.backward()
        self.optimizer_d.step()

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
