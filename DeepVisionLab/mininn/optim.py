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