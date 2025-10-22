# diffusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# beta scheduler
class BetaScheduler:
    def __init__(self, T, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    

# diffusion
class GaussianDiffusion:
    def __init__(self, model, scheduler: BetaScheduler, device="cuda", **kwargs):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.model_pred = kwargs.get("model_pred", "eps")

    # q(x_t | x_0): forward
    def q_sample(self, x0, t, noise=None):
        """
        input:
            x0: [B, C, H, W] as example
            t: [B,]
        output:
            x_t: [B, C, H, W] as example

        x_t = sqrt(alphas_cumprod[t]) * x0 + sqrt(1 - alphas_cumprod[t]) * noise
        """
        if noise is None:
            noise = torch.randn_like(x0) # generate standard normal noise with the same shape as x0
        
        # compute alphas_cumprod_t 
        alphas_cumprod_t = self.scheduler.alphas_cumprod[t] # [B,]
        alphas_cumprod_t = alphas_cumprod_t.view(-1, *([1]*(x0.dim()-1))) # eg:[B, 1, 1, 1]

        x_t = torch.sqrt(alphas_cumprod_t)*x0 + torch.sqrt(1-alphas_cumprod_t)*noise
        return x_t
    
    # p_theta(x_{t-1} | x_t): backward
    def p_sample(self, x_t, t, cond):
        """
        input:
            x_t: [B, C, H, W] as example
            t: [B,]
        output:
            x_t-1: [B, C, H, W] as example

        x_{t-1} = 1/sqrt(alphas_t) * (x_t - (1-alphas_t)/sqrt(1-alphas_bar_t) * eps_theta) + sigma_t * z
        """
        # compute alphas_cumprod_t, alphas_t and betas_t
        alphas_cumprod_t = self.scheduler.alphas_cumprod[t].view(-1, *([1]*(x_t.dim()-1)))
        alphas_t = self.scheduler.alphas[t].view(-1, *([1]*(x_t.dim()-1)))
        betas_t = self.scheduler.betas[t].view(-1, *([1]*(x_t.dim()-1)))

        # predict noise
        if self.model_pred == "eps":
            eps_theta = self.model(x_t, t, cond)
        # predict x0
        if self.model_pred == "x0":
            x0_pred = self.model(x_t, t, cond)
            eps_theta = (x_t - torch.sqrt(alphas_cumprod_t)*x0_pred) / torch.sqrt(1 - alphas_cumprod_t)
        else:
            raise ValueError(f"Unknown model_pred type: {self.model_pred}")

        # mu
        mu = (1 / torch.sqrt(alphas_t)) * (x_t - ((1-alphas_t)/torch.sqrt(1-alphas_cumprod_t))*eps_theta)

        # standard deviation
        sigma_t = torch.sqrt(betas_t)

        # sample random noise z
        z = torch.randn_like(x_t)
        mask = (t==0).view(-1,1,1,1)
        z = z * (~mask) # No noise is added at (t = 0)

        # sample x_t-1 (named x_pre) 
        x_pre = mu + sigma_t * z

        return x_pre
    
    # sample from noise
    def sample(self, shape, cond):
        """
        input:
            shape: [B, C, H, W] as example
        output:
            x0: [B, C, H, W]
        """
        device = self.device
        T = self.scheduler.T

        # sample noise
        x_T = torch.randn(shape, device=device)

        # backward from T to 0
        with torch.no_grad():
            x_t = x_T
            for t in reversed(range(T)):
                t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
                x_t = self.p_sample(x_t, t_tensor, cond)

        return x_t
    
    # Noise prediction loss during training
    def training_loss(self, x0, cond):
        """
        input:
            x0: [B, C, H, W] as example
        output:
            loss: [B,]

        loss = mse(noise_pred, noise)
        """
        device = x0.device

        # sample t
        t = torch.randint(0, self.scheduler.T, (x0.shape[0],), device=device)

        # sample noise
        noise = torch.randn_like(x0)

        # compute x_t
        x_t = self.q_sample(x0, t, noise)

        # predict eps_theta
        if self.model_pred == "eps": 
            eps_theta = self.model(x_t, t, cond)
            loss = F.mse_loss(eps_theta, noise)
        # predict x0
        if self.model_pred == "x0": 
            x0_pred = self.model(x_t, t, cond)
            loss = F.mse_loss(x0_pred, x0)
        else:
            raise ValueError(f"Unknown model_pred type: {self.model_pred}")

        return loss


