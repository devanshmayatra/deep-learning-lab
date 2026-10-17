import torch

class Adam:
    def __init__(self, parameters, learning_rate=0.001):
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.t = 0

        self.first_moments = [
            torch.zeros_like(p) for p in self.parameters
        ]
        self.second_moments = [
            torch.zeros_like(p) for p in self.parameters
        ]
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = None
            
    def step(self):
        with torch.no_grad():
            self.t += 1

            for p, m, v in zip(
                self.parameters,
                self.first_moments,
                self.second_moments
            ):
                if p.grad is None:
                    continue

                m.mul_(0.9).add_(0.1 * p.grad)
                v.mul_(0.999).add_(0.001 * p.grad.square())

                m_hat = m / (1 - 0.9**self.t)
                v_hat = v / (1 - 0.999**self.t)

                update = m_hat / (v_hat.sqrt() + 1e-8)
                p.add_(update, alpha=-self.learning_rate)
                
    def state_dict(self):
        return {
            "learning_rate": self.learning_rate,
            "t": self.t,
            "first_moments": [
                m.detach().clone() for m in self.first_moments
            ],
            "second_moments": [
                v.detach().clone() for v in self.second_moments
            ],
        }
    
    def load_state_dict(self, state):
        expected_keys = {
            "learning_rate", "t",
            "first_moments", "second_moments"
        }

        if set(state) != expected_keys:
            raise ValueError("Optimizer state keys do not match.")

        for name in ("first_moments", "second_moments"):
            saved = state[name]

            if len(saved) != len(self.parameters):
                raise ValueError(f"Parameter count mismatch for {name}.")

            for p, tensor in zip(self.parameters, saved):
                if tensor.shape != p.shape:
                    raise ValueError(f"Shape mismatch in {name}.")

        if not isinstance(state["t"], int) or state["t"] < 0:
            raise ValueError("Step counter must be a non-negative integer.")

        with torch.no_grad():
            for current, saved in zip(
                self.first_moments, state["first_moments"]
            ):
                current.copy_(saved)

            for current, saved in zip(
                self.second_moments, state["second_moments"]
            ):
                current.copy_(saved)

        self.learning_rate = state["learning_rate"]
        self.t = state["t"]
        
class AdamW(Adam):
    def __init__(
        self,
        parameters,
        learning_rate=0.001,
        weight_decay=0.01,
    ):
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")

        super().__init__(parameters, learning_rate)
        self.weight_decay = weight_decay

    def step(self):
        # Apply decay separately from Adam's moment calculations.
        with torch.no_grad():
            for p in self.parameters:
                if p.grad is not None:
                    p.mul_(
                        1 - self.learning_rate * self.weight_decay
                    )

        super().step()

    def state_dict(self):
        state = super().state_dict()
        state["weight_decay"] = self.weight_decay
        return state

    def load_state_dict(self, state):
        expected_keys = {
            "learning_rate",
            "weight_decay",
            "t",
            "first_moments",
            "second_moments",
        }

        if set(state) != expected_keys:
            raise ValueError("AdamW state keys do not match.")

        if state["weight_decay"] < 0:
            raise ValueError("weight_decay must be non-negative.")

        # Pass Adam only the keys it expects.
        adam_state = {
            key: value
            for key, value in state.items()
            if key != "weight_decay"
        }
        super().load_state_dict(adam_state)
        self.weight_decay = state["weight_decay"]