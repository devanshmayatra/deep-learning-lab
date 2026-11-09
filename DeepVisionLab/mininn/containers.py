from .layers import Conv2D, BatchNorm2D, ReLU, MaxPool2D

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
        self, in_channels, out_channels=None, stride=1,
        use_skip=True, *, device="cpu", generator=None
    ):
        if out_channels is None:
            out_channels = in_channels

        self.use_skip = use_skip

        self.branch = Sequential([
            Conv2D(
                in_channels, out_channels, 3,
                stride=stride, padding=1,
                device=device, generator=generator
            ),
            BatchNorm2D(out_channels, device=device),
            ReLU(),
            Conv2D(
                out_channels, out_channels, 3,
                stride=1, padding=1,
                device=device, generator=generator
            ),
            BatchNorm2D(out_channels, device=device),
        ])

        self.shortcut = None

        if self.use_skip and (
            in_channels != out_channels or stride != 1
        ):
            self.shortcut = Conv2D(
                in_channels, out_channels,
                kernel_size=1, stride=stride,
                device=device, generator=generator
            )

    def __call__(self, x):
        residual = self.branch(x)

        if self.use_skip:
            shortcut = x if self.shortcut is None else self.shortcut(x)
            self.out = residual + shortcut
        else:
            self.out = residual

        return self.out
    
    def train(self):
        self.branch.train()

    def eval(self):
        self.branch.eval()
        
    def state_dict(self):
        state = {
            f"branch.{name}": tensor
            for name, tensor in self.branch.state_dict().items()
        }

        if self.shortcut is not None:
            state.update({
                f"shortcut.{name}": tensor
                for name, tensor in self.shortcut.state_dict().items()
            })

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

        branch_state = {
            name[len("branch."):]: tensor
            for name, tensor in state.items()
            if name.startswith("branch.")
        }
        self.branch.load_state_dict(branch_state)

        if self.shortcut is not None:
            shortcut_state = {
                name[len("shortcut."):]: tensor
                for name, tensor in state.items()
                if name.startswith("shortcut.")
            }
            self.shortcut.load_state_dict(shortcut_state)

    def parameters(self):
        params = self.branch.parameters()

        if self.shortcut is not None:
            params += self.shortcut.parameters()

        return params

class Inception:
    def __init__(self, in_channels, *, device="cpu", generator=None):

        a = Conv2D(in_channels=in_channels, out_channels=16, kernel_size=1, device=device, generator=g)
        a_act = ReLU()
        
        self.branch_a = Sequential([a,a_act])
        
        b_1 = Conv2D(in_channels=in_channels, out_channels=16, kernel_size=1, device=device, generator=g)
        b1_act = ReLU()
        
        b = Conv2D(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1, device=device, generator=g)
        b_act = ReLU()
        
        self.branch_b = Sequential([b_1, b1_act, b, b_act])
        
        c_1 = Conv2D(in_channels=in_channels, out_channels=8, kernel_size=1, device=device, generator=g)
        c1_act = ReLU()
        
        c = Conv2D(in_channels=8, out_channels=12, kernel_size=5, stride=1, padding=2, device=device, generator=g)
        c_act = ReLU()
        
        self.branch_c = Sequential([c_1, c1_act, c, c_act])
        
        d = MaxPool2D(3,1,1)
        d_1 = Conv2D(in_channels=in_channels, out_channels=8, kernel_size=1, device=device, generator=g)
        d_act = ReLU()
        
        self.branch_d = Sequential([d, d_1, d_act])
    
    def __call__(self, x):
        
        output_a = self.branch_a(x)
        output_b = self.branch_b(x)
        output_c = self.branch_c(x)
        output_d = self.branch_d(x)
        
        output = torch.cat(
            (output_a, output_b, output_c, output_d),
            dim=1,
        )
        return output
    
    def parameters(self):

        return (
            self.branch_a.parameters()
            + self.branch_b.parameters()
            + self.branch_c.parameters()
            + self.branch_d.parameters()
        )
    
    def train(self):
        for branch in (
            self.branch_a, self.branch_b,
            self.branch_c, self.branch_d,
        ):
            branch.train()


    def eval(self):
        for branch in (
            self.branch_a, self.branch_b,
            self.branch_c, self.branch_d,
        ):
            branch.eval()


    def state_dict(self):
        state = {}

        for name in ("branch_a", "branch_b", "branch_c", "branch_d"):
            branch = getattr(self, name)

            for key, tensor in branch.state_dict().items():
                state[f"{name}.{key}"] = tensor

        return state


    def load_state_dict(self, state):
        expected = self.state_dict()

        # Validate every key and shape before changing parameters.
        if set(state) != set(expected):
            missing = set(expected) - set(state)
            unexpected = set(state) - set(expected)
            raise ValueError(
                f"Missing keys: {sorted(missing)}; "
                f"Unexpected keys: {sorted(unexpected)}"
            )

        for key, tensor in expected.items():
            if state[key].shape != tensor.shape:
                raise ValueError(f"Shape mismatch for {key}.")

        for name in ("branch_a", "branch_b", "branch_c", "branch_d"):
            prefix = f"{name}."
            branch_state = {
                key[len(prefix):]: tensor
                for key, tensor in state.items()
                if key.startswith(prefix)
            }
            getattr(self, name).load_state_dict(branch_state)