import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm


class StableDiffusionTrainingPipeline:
    def __init__(self, model, scheduler, train_dataset, batch_size=16, lr=1e-4, device="cuda"):
        self.model = model
        self.scheduler = scheduler
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.device = device
        self.loss_fn = torch.nn.MSELoss()

        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()

    def train(self, epochs, save_dir="checkpoints"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self.model.train()
            total_loss = 0.0

            for batch_idx, (x, _) in enumerate(tqdm(self.train_dataloader, desc="Training")):
                x = x.unsqueeze(1)  # Add channel dimension

                if self.device == "cuda" and torch.cuda.is_available():
                    x = x.to(self.device)

                noise = torch.randn_like(x).to(self.device)
                timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (x.size(0),), device=self.device)
                noisy_x = self.scheduler.add_noise(x, noise, timesteps)

                predicted_noise = self.model(noisy_x, timesteps).sample
                loss = self.loss_fn(predicted_noise, noise)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}/{len(self.train_dataloader)} - Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}.")
