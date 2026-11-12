from .layers import Conv2D, BatchNorm2D, ReLU, MaxPool2D, ConvSwiGLU, GroupNorm2D
import torch

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
        self, in_channels, out_channels=None, stride=1, activation=ReLU, use_swiglu=False,
        use_skip=True, *, device="cpu", generator=None
    ):
        if out_channels is None:
            out_channels = in_channels

        self.use_skip = use_skip
        self.final_relu = activation()
        
        middle_activation = (
            ConvSwiGLU(
                out_channels, out_channels,
                device=device, generator=generator,
            )
            if use_swiglu
            else activation()
        )

        if use_swiglu:
            norm_before = GroupNorm2D(
                out_channels, groups=8, device=device
            )
            norm_after = GroupNorm2D(
                out_channels, groups=8, device=device
            )
        else:
            norm_before = BatchNorm2D(out_channels, device=device)

        branch_layers = [
            Conv2D(
                in_channels, out_channels, 3,
                stride=stride, padding=1,
                device=device, generator=generator,
            ),
            norm_before,
            middle_activation,
        ]

        if use_swiglu:
            branch_layers.append(norm_after)

        branch_layers.extend([
            Conv2D(
                out_channels, out_channels, 3,
                padding=1,
                device=device, generator=generator,
            ),
            BatchNorm2D(out_channels, device=device),
        ])

        self.branch = Sequential(branch_layers)

        self.shortcut = None

        if self.use_skip and (
            in_channels != out_channels or stride != 1
        ):
            self.shortcut = Sequential([Conv2D(
                in_channels, out_channels,
                kernel_size=1, stride=stride,
                device=device, generator=generator
            ),
            BatchNorm2D(out_channels, device=device),
            ])

    def __call__(self, x):
        residual = self.branch(x)

        if self.use_skip:
            shortcut = x if self.shortcut is None else self.shortcut(x)
            self.out = residual + shortcut
        else:
            self.out = residual

        self.out = self.final_relu(self.out)
        return self.out
    
    def train(self):
        self.branch.train()
        if self.shortcut is not None:
            self.shortcut.train()

    def eval(self):
        self.branch.eval()
        if self.shortcut is not None:
            self.shortcut.eval()
        
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
    def __init__(self, in_channels, ch1, ch3_reduce, ch3, ch5_reduce, ch5, pool_proj, *, device="cpu", generator=None):

        a = Conv2D(in_channels=in_channels, out_channels=ch1, kernel_size=1, device=device, generator=generator)
        a_act = ReLU()
        
        self.branch_a = Sequential(
            [a, 
            #   BatchNorm2D(ch1, device=device), 
                a_act
                ]
            )
        
        b_1 = Conv2D(in_channels=in_channels, out_channels=ch3_reduce, kernel_size=1, device=device, generator=generator)
        b1_act = ReLU()
        
        b = Conv2D(in_channels=ch3_reduce, out_channels=ch3, kernel_size=3, stride=1, padding=1, device=device, generator=generator)
        b_act = ReLU()
        
        self.branch_b = Sequential([b_1, 
                                    # BatchNorm2D(ch3_reduce, device=device), 
                                    b1_act, b, 
                                    # BatchNorm2D(ch3, device=device), 
                                    b_act])
        
        c_1 = Conv2D(in_channels=in_channels, out_channels=ch5_reduce, kernel_size=1, device=device, generator=generator)
        c1_act = ReLU()
        
        c = Conv2D(in_channels=ch5_reduce, out_channels=ch5, kernel_size=5, stride=1, padding=2, device=device, generator=generator)
        c_act = ReLU()
        
        self.branch_c = Sequential([c_1, 
                                    # BatchNorm2D(ch5_reduce, device=device), 
                                    c1_act, c, 
                                    # BatchNorm2D(ch5, device=device), 
                                    c_act])
        
        d = MaxPool2D(3,1,1)
        d_1 = Conv2D(in_channels=in_channels, out_channels=pool_proj, kernel_size=1, device=device, generator=generator)
        d_act = ReLU()
        
        self.branch_d = Sequential([d, d_1, 
                                    # BatchNorm2D(pool_proj, device=device), 
                                    d_act])
        self.out_channels = ch1 + ch3 + ch5 + pool_proj
    
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
            
class InceptionWBatchNorm2D:
    def __init__(self, in_channels, ch1, ch3_reduce, ch3, ch5_reduce, ch5, pool_proj, *, device="cpu", generator=None):

        a = Conv2D(in_channels=in_channels, out_channels=ch1, kernel_size=1, device=device, generator=generator)
        a_act = ReLU()
        
        self.branch_a = Sequential(
            [a, 
              BatchNorm2D(ch1, device=device), 
                a_act
                ]
            )
        
        b_1 = Conv2D(in_channels=in_channels, out_channels=ch3_reduce, kernel_size=1, device=device, generator=generator)
        b1_act = ReLU()
        
        b = Conv2D(in_channels=ch3_reduce, out_channels=ch3, kernel_size=3, stride=1, padding=1, device=device, generator=generator)
        b_act = ReLU()
        
        self.branch_b = Sequential([b_1, 
                                    BatchNorm2D(ch3_reduce, device=device), 
                                    b1_act, b, 
                                    BatchNorm2D(ch3, device=device), 
                                    b_act])
        
        c_1 = Conv2D(in_channels=in_channels, out_channels=ch5_reduce, kernel_size=1, device=device, generator=generator)
        c1_act = ReLU()
        
        c = Conv2D(in_channels=ch5_reduce, out_channels=ch5, kernel_size=5, stride=1, padding=2, device=device, generator=generator)
        c_act = ReLU()
        
        self.branch_c = Sequential([c_1, 
                                    BatchNorm2D(ch5_reduce, device=device), 
                                    c1_act, c, 
                                    BatchNorm2D(ch5, device=device), 
                                    c_act])
        
        d = MaxPool2D(3,1,1)
        d_1 = Conv2D(in_channels=in_channels, out_channels=pool_proj, kernel_size=1, device=device, generator=generator)
        d_act = ReLU()
        
        self.branch_d = Sequential([d, d_1, 
                                    BatchNorm2D(pool_proj, device=device), 
                                    d_act])
        self.out_channels = ch1 + ch3 + ch5 + pool_proj
    
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