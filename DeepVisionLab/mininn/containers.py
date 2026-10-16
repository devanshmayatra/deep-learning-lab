from .layers import Conv2D, BatchNorm2D, ReLU

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def train(self):
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()
            elif hasattr(layer, "training"):
                layer.training = True

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()
            elif hasattr(layer, "training"):
                layer.training = False
                
    
    def state_dict(self):
        state = {}

        for i, layer in enumerate(self.layers):
            for name, tensor in layer.state_dict().items():
                state[f"layers.{i}.{name}"] = tensor

        return state
    
    def load_state_dict(self, state):
        expected = self.state_dict()

        if set(state) != set(expected):
            missing = set(expected) - set(state)
            unexpected = set(state) - set(expected)
            raise ValueError(
                f"Missing keys: {sorted(missing)}; "
                f"Unexpected keys: {sorted(unexpected)}"
            )

        for name, tensor in expected.items():
            if state[name].shape != tensor.shape:
                raise ValueError(f"Shape mismatch for {name}.")

        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}."

            layer_state = {
                name[len(prefix):]: tensor
                for name, tensor in state.items()
                if name.startswith(prefix)
            }

            layer.load_state_dict(layer_state)
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    
class ResidualBlock:
    def __init__(
        self, channels, use_skip=True, *,
        device="cpu", generator=None
    ):
        self.use_skip = use_skip

        self.branch = Sequential([
            Conv2D(
                channels, channels, 3, padding=1,
                device=device, generator=generator
            ),
            BatchNorm2D(channels, device=device),
            ReLU(),
            Conv2D(
                channels, channels, 3, padding=1,
                device=device, generator=generator
            ),
            BatchNorm2D(channels, device=device),
        ])

    def __call__(self, x):
        residual = self.branch(x)
        self.out = residual + x if self.use_skip else residual
        return self.out
    
    def train(self):
        self.branch.train()

    def eval(self):
        self.branch.eval()
        
    def state_dict(self):
        return {
            f"branch.{name}": tensor
            for name, tensor in self.branch.state_dict().items()
        }
        
    def load_state_dict(self, state):
        prefix = "branch."

        if any(not name.startswith(prefix) for name in state):
            raise ValueError("ResidualBlock expects branch-prefixed keys.")

        branch_state = {
            name[len(prefix):]: tensor
            for name, tensor in state.items()
        }

        self.branch.load_state_dict(branch_state)

    def parameters(self):
        return self.branch.parameters()