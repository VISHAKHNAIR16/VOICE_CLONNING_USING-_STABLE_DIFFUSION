import torch

class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x, noise, timesteps):
        """
        Adds noise to the input `x` based on the DDPM schedule.
        """
    # Ensure alpha_cumprod is on the same device as timesteps
        alpha_t = self.alpha_cumprod.to(timesteps.device)[timesteps].view(-1, 1, 1, 1)
        alpha_t = alpha_t.to(x.device)  # Move to the same device as `x`

        return torch.sqrt(alpha_t) * x + torch.sqrt(1.0 - alpha_t) * noise


    def remove_noise(self, noisy_x, predicted_noise, timesteps):
        """
        Removes noise from the noisy input using predicted noise.
        """
        # Ensure `self.alpha_cumprod` and `self.betas` are on the same device as `timesteps`
        alpha_t = self.alpha_cumprod.to(timesteps.device)[timesteps].view(-1, 1, 1, 1).to(noisy_x.device)
        beta_t = self.betas.to(timesteps.device)[timesteps].view(-1, 1, 1, 1).to(noisy_x.device)

        # Adjust dimensions to match noisy_x and predicted_noise
        if noisy_x.shape != predicted_noise.shape:
            predicted_noise = self._adjust_tensor_size(noisy_x, predicted_noise)

        return torch.sqrt(1.0 / alpha_t) * (
            noisy_x - beta_t / torch.sqrt(1.0 - alpha_t) * predicted_noise
        )


    def _adjust_tensor_size(self, tensor1, tensor2):
        """
        Adjusts the size of tensor2 to match tensor1.
        """
        _, _, h1, w1 = tensor1.shape
        _, _, h2, w2 = tensor2.shape

        # Adjust height
        if h1 > h2:
            tensor2 = torch.nn.functional.pad(tensor2, (0, 0, 0, h1 - h2))
        elif h1 < h2:
            tensor2 = tensor2[:, :, :h1, :]

        # Adjust width
        if w1 > w2:
            tensor2 = torch.nn.functional.pad(tensor2, (0, w1 - w2, 0, 0))
        elif w1 < w2:
            tensor2 = tensor2[:, :, :, :w1]

        return tensor2
