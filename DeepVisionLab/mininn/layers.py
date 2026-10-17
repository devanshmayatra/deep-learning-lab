import torch

class Conv2D:  
    def __init__(
    self, in_channels, out_channels, kernel_size,
    stride=1, padding=0, *, device="cpu", generator=None
    ):
        fan_in = in_channels * kernel_size * kernel_size

        self.weight = torch.randn(
            (out_channels, in_channels, kernel_size, kernel_size),
            generator=generator,
            device=device
        ) * (2 / fan_in)**0.5

        self.bias = torch.zeros(out_channels, device=device)
        self.stride = stride
        self.padding = padding

        self.weight.requires_grad_(True)
        self.bias.requires_grad_(True)
    
    def __call__(self, x):
        k = self.weight.shape[2]

        patches = torch.nn.functional.unfold(
            x, kernel_size=k, padding=self.padding, stride=self.stride
        )
        weights_flat = self.weight.reshape(self.weight.shape[0], -1)

        out = weights_flat @ patches
        out = out + self.bias.reshape(1, -1, 1)
        
        height_out = (x.shape[2] + 2 * self.padding - k) // self.stride + 1
        width_out = (x.shape[3] + 2 * self.padding - k) // self.stride + 1

        self.out = out.reshape(
            x.shape[0], self.weight.shape[0], height_out, width_out
        )
        return self.out
    
    def state_dict(self):
        return {
            "weight": self.weight.detach().clone(),
            "bias": self.bias.detach().clone(),
        }
        
    def load_state_dict(self, state):
        names = ["weight"]
        if self.bias is not None:
            names.append("bias")

        if set(state) != set(names):
            raise ValueError("State keys do not match this layer.")

        for name in names:
            if state[name].shape != getattr(self, name).shape:
                raise ValueError(f"Shape mismatch for {name}.")

        with torch.no_grad():
            for name in names:
                getattr(self, name).copy_(state[name])
    
    def parameters(self):
        return [self.weight] + [self.bias]
    
class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self,x):
        # (B, C * k * k, number_of_patches)
        patches = torch.nn.functional.unfold(
            x, kernel_size=self.kernel_size, stride=self.stride
        )
        
        B, C, H, W = x.shape
        k = self.kernel_size

        patches = patches.reshape(B, C, k * k, -1)
        
        out = patches.max(dim=2).values
        
        height_out = (H - k) // self.stride + 1
        width_out = (W - k) // self.stride + 1

        self.out = out.reshape(B, C, height_out, width_out)
        return self.out

    def state_dict(self):
        return {}
    
    def load_state_dict(self, state):
        if state:
            raise ValueError("This layer expects an empty state dictionary.")

    def parameters(self):
        return []

class Flatten:
    def __call__(self, x):
        self.out = x.reshape(x.shape[0], -1)
        return self.out
    
    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError("This layer expects an empty state dictionary.")

    def parameters(self):
        return []

class Linear:
    
    def __init__(
        self, fan_in, fan_out, bias=True, *,
        device="cpu", generator=None
    ):
        self.weight = torch.randn(
            (fan_in, fan_out),
            generator=generator,
            device=device
        ) * (2 / fan_in)**0.5

        self.bias = (
            torch.zeros(fan_out, device=device)
            if bias else None
        )

        self.weight.requires_grad_(True)
        if self.bias is not None:
            self.bias.requires_grad_(True)
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def state_dict(self):
        state = {
            "weight": self.weight.detach().clone()
        }

        if self.bias is not None:
            state["bias"] = self.bias.detach().clone()

        return state
    
    def load_state_dict(self, state):
        names = ["weight"]
        if self.bias is not None:
            names.append("bias")

        if set(state) != set(names):
            raise ValueError("State keys do not match this layer.")

        for name in names:
            if state[name].shape != getattr(self, name).shape:
                raise ValueError(f"Shape mismatch for {name}.")

        with torch.no_grad():
            for name in names:
                getattr(self, name).copy_(state[name])
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
class ReLU:
    def __call__(self, x):
        self.out = torch.clamp(x, min=0)
        return self.out
    
    def state_dict(self):
        return {}
    
    def load_state_dict(self, state):
        if state:
            raise ValueError("This layer expects an empty state dictionary.")
    
    def parameters(self):
        return []

class BatchNorm2D:
    def __init__(
        self, channels, eps=1e-5, training=True,
        momentum=0.1, *, device="cpu"
    ):
        self.eps = eps
        self.training = training
        self.momentum = momentum

        shape = (1, channels, 1, 1)

        self.running_mean = torch.zeros(shape, device=device)
        self.running_var = torch.ones(shape, device=device)

        self.gamma = torch.ones(
            shape, device=device, requires_grad=True
        )
        self.beta = torch.zeros(
            shape, device=device, requires_grad=True
        )
    
    def __call__(self, x):
        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

            # Number of values contributing to each channel's statistics
            n = x.shape[0] * x.shape[2] * x.shape[3]
            if n <= 1:
                raise ValueError("Training requires more than one value per channel.")

            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)

                # Match PyTorch's convention for the running variance
                unbiased_var = var * n / (n - 1)
                self.running_var.lerp_(unbiased_var, self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        self.out = self.gamma * x_hat + self.beta
        return self.out

    def state_dict(self):
        return {
            "gamma": self.gamma.detach().clone(),
            "beta": self.beta.detach().clone(),
            "running_mean": self.running_mean.detach().clone(),
            "running_var": self.running_var.detach().clone(),
        }
        
    def load_state_dict(self, state):
        names = ("gamma", "beta", "running_mean", "running_var")

        # Validate everything before changing the layer
        if set(state) != set(names):
            raise ValueError("State keys do not match BatchNorm2D.")

        for name in names:
            if state[name].shape != getattr(self, name).shape:
                raise ValueError(f"Shape mismatch for {name}.")

        with torch.no_grad():
            for name in names:
                getattr(self, name).copy_(state[name])
    
    def parameters(self):
        return [self.gamma, self.beta]
    

class GlobalAvgPool2D:
    def __call__(self, x):
        self.out = x.mean(dim=(2, 3))
        return self.out

    def parameters(self):
        return []

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError("This layer expects an empty state dictionary.")


class Dropout:
    def __init__(self, p=0.5):
        if not 0 <= p < 1:
            raise ValueError("p must satisfy 0 <= p < 1.")

        self.p = p
        self.training = True

    def __call__(self, x):
        if not self.training or self.p == 0:
            self.out = x
        else:
            keep_prob = 1 - self.p
            mask = torch.rand_like(x) < keep_prob
            self.out = x * mask / keep_prob

        return self.out

    def parameters(self):
        return []

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError("Dropout expects an empty state dictionary.")