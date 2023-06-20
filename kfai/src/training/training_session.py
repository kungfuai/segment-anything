import os
from kfai.src.data.DatasetBuilder import DatasetBuilder
from kfai.segment_anything.predictor import SamPredictor
from kfai.segment_anything import sam_model_registry
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
import torch
from trainer import Trainer
from training_config import TrainingConfig


class TrainingSession:
    """TrainingSession is responsible for model training setup and configuration."""

    def __init__(self, config):
        self.config = config

    def run(self):
        self.create_datasets()
        self.create_dataloaders()
        self.create_predictor()
        self.configure_device()
        self.configure_predictor()        
        self.create_optimizer()
        self.create_criterion()
        self.create_trainer()
        self.trainer.run()

    def create_datasets(self):
        self.dataset = DatasetBuilder(root_dir=self.config.root_dir, mask_dir=self.config.mask_dir)

    def create_dataloaders(self):
        train_split = int(self.dataset.__len__() * self.config.train_split)
        test_split = self.dataset.__len__() - train_split
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [train_split, test_split])

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=True)

    def create_predictor(self):
        download_url('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', 'kfai')
        checkpoint = 'kfai/sam_vit_b_01ec64.pth'
        sam = sam_model_registry['vit_b'](checkpoint=checkpoint)
        self.predictor = SamPredictor(sam)

    def configure_predictor(self):
        self.predictor.model.to(self.device)

        for name, param in self.predictor.model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

    def configure_device(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def create_optimizer(self):
        self.optimizer = torch.optim.Adam(self.predictor.model.mask_decoder.parameters(), lr = self.config.learning_rate)

    def create_criterion(self):
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.config.pos_weight]).to(self.device))

    def create_trainer(self):
        self.trainer = Trainer(
            predictor=self.predictor,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device = self.device,
            config=self.config,
        )


def main():
    config = TrainingConfig.parse_args()
    session = TrainingSession(config)
    session.run()


if __name__ == "__main__":
    main()
