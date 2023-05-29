from __future__ import division, print_function, absolute_import

import wandb

from torchreid import metrics
from torchreid.losses import TripletLoss, CrossEntropyLoss
from pytorch_metric_learning.losses import ArcFaceLoss

import torch

from ..engine import Engine


class ImageTripletArcFaceEngine(Engine):
    r"""Triplet-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_t=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        margin=0.3,
        weight_t=1,
        weight_x=1,
        scheduler=None,
        use_gpu=True
    ):
        super(ImageTripletArcFaceEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        assert weight_t >= 0 and weight_x >= 0
        assert weight_t + weight_x > 0
        self.weight_t = weight_t
        self.weight_x = weight_x

        # determine embedding dimensionality using dummy forward pass
        input_dummy = torch.zeros(1, 3, datamanager.width, datamanager.height)
        if self.use_gpu:
            input_dummy = input_dummy.to(f"cuda:{wandb.config.gpu_device}")

        with torch.no_grad():
            # set model to evaluation mode before forwarding
            self.model.eval()
            features = self.model(input_dummy)
        self.feat_dim = features.shape[1]

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = ArcFaceLoss(
            num_classes=self.datamanager.num_train_pids,
            embedding_size=self.feat_dim,
        )
        # append ArcFaceLoss Parameters to optimizer
        self.optimizer.add_param_group({'params': self.criterion_x.parameters()})

        # self.criterion_x = CrossEntropyLoss(
        #     num_classes=self.datamanager.num_train_pids,
        #     use_gpu=self.use_gpu,
        #     label_smooth=label_smooth
        # )

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        # Preview images in the batch
        # # Convert tensors to to numpy arrays
        # imgs_np = imgs.cpu().numpy()

        # # Convert 3 channel images: (3, 256, 128) -> (256, 128, 3)
        # imgs_np = imgs_np.transpose(0, 2, 3, 1)

        # import matplotlib.pyplot as plt

        # # Plot the first image in the batch
        # plt.imshow(imgs_np[0])
        # plt.show()

        if self.use_gpu:
            imgs = imgs.to(f"cuda:{wandb.config.gpu_device}")
            pids = pids.to(f"cuda:{wandb.config.gpu_device}")

        outputs, features = self.model(imgs)

        loss = 0
        loss_summary = {}

        if self.weight_t > 0:
            loss_t = self.compute_loss(self.criterion_t, features, pids)
            loss += self.weight_t * loss_t
            loss_summary['loss_t'] = loss_t.item()

        if self.weight_x > 0:
            loss_x = self.compute_loss(self.criterion_x, features, pids)
            loss += self.weight_x * loss_x
            loss_summary['loss_x'] = loss_x.item()
            loss_summary['acc'] = metrics.accuracy(outputs, pids)[0].item()

        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary
