import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch


class Trainer:
    """Trainer is responsible for model training."""

    def __init__(
        self,
        predictor,
        train_dataset,
        val_dataset,
        train_dataloader,
        val_dataloader,
        optimizer,
        criterion,
        device,
        config,
    ):
        self.predictor = predictor
        self.train_dataset=train_dataset,
        self.val_dataset=val_dataset,
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = config.epochs
        self.device = device

    def run(self):
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()

            print(f"EPOCH {epoch+1}")
            print(f"Training Loss: {train_loss}")
            print(f"Validation Loss: {val_loss}")

        #self.visualize()

    def train_epoch(self):
        self.predictor.model.train()
        epoch_training_loss = 0.0

        for i, data in tqdm(enumerate(self.train_dataloader, 0), total=len(self.train_dataloader)):
            inputs, labels = data
            labels = labels.to(self.device)/255

            inputs = inputs.to(self.device)

            self.optimizer.zero_grad()

            self.predictor.set_image(np.asarray(inputs[0].cpu()))
            masks, scores, logits = self.predictor.predict(
                multimask_output=False,
                return_logits=True
            )

            loss = self.criterion(masks[0], labels.float())
            loss.backward()
            self.optimizer.step()

            epoch_training_loss += loss.item()
        
        return epoch_training_loss/len(self.train_dataloader)

    def validate_epoch(self):
        self.predictor.model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for i, data in tqdm(enumerate(self.val_dataloader, 0), total=len(self.val_dataloader)):
                inputs, labels = data
                labels = labels.to(self.device)/255

                inputs = inputs.to(self.device)

                self.predictor.set_image(np.asarray(inputs[0].cpu()))
                masks, scores, logits = self.predictor.predict(
                    multimask_output=False,
                    return_logits=True
                )
                
                loss = self.criterion(masks[0], labels.float())
                epoch_val_loss += loss.item()

        return epoch_val_loss/len(self.val_dataloader)
    

    def visualize(self):
        for i in range(self.val_dataset.__len__()):
            with torch.no_grad():
                item = self.val_dataset.__getitem__(i)[0][0]
                print(item.shape)
                self.predictor.set_image(item)
                mask, score, logit = self.predictor.predict(
                    multimask_output=False,
                    return_logits=True
                )

            plt.figure(figsize=(3,3))
            output = np.asarray(mask.squeeze().detach().cpu())
            plt.imshow(output)

            plt.figure(figsize=(3,3))
            label = np.asarray(self.val_dataset.__getitem__(i)[0][1])
            plt.imshow(label)