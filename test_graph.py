from time import time
from sinabs.exodus.spike import IntegrateAndFireCapturable
from sinabs.activation import PeriodicExponential
import torch

class Network(torch.nn.Module):
    def __init__(self, c_in: int, c_out: int, alpha: torch.Tensor):
        super().__init__()
        self.weights = torch.nn.Linear(c_in, c_out, bias=False)
        self.register_parameter("alpha", torch.nn.Parameter(alpha))
        self.register_buffer("threshold", torch.ones_like(alpha))
        self.register_buffer("v_mem_init", torch.zeros_like(alpha))
        self.surrogate_grad_fn = PeriodicExponential()

    def forward_inner(self, x):
        return IntegrateAndFireCapturable.apply(
            x,
            self.alpha,
            self.v_mem_init,
            self.threshold,
            self.threshold,
            -self.threshold,
            self.surrogate_grad_fn,
            None
        )

    def forward(self, x):
        x = self.weights(x)
        batch_size, num_timesteps, c_out = x.shape
        x = x.movedim(1, -1).flatten(0, 1)
        spikes, v_mem = self.forward_inner(x)
        v_mem = v_mem - self.threshold.unsqueeze(1) * spikes
        # self.v_mem_init = v_mem[:, -1].contiguous().detach()

        return spikes.reshape(batch_size, c_out, num_timesteps).movedim(-1, 1)


batch_size = 100
c_out = 500
c_in = 300
num_timesteps = 500
alpha = torch.rand(batch_size * c_out, requires_grad=True)
graphed_alpha = alpha.clone().detach()

network = Network(c_in, c_out, alpha).cuda()
graphed_network = Network(c_in, c_out, graphed_alpha).cuda()
graphed_network.weights.weight.data = network.weights.weight.data.clone().detach()

optimizer = torch.optim.SGD(network.parameters(), lr=0.001)
graphed_optimizer = torch.optim.SGD(graphed_network.parameters(), lr=0.001)

initial_input = torch.randn(batch_size, num_timesteps, c_in, device="cuda")
graphed_network = torch.cuda.make_graphed_callables(graphed_network, (initial_input,))

real_inputs = [torch.randn_like(initial_input) for _ in range(20)]
graphed_real_inputs = [x.clone() for x in real_inputs]

t0 = time()
for inp in real_inputs:
    optimizer.zero_grad()
    output = network(inp)
    print(s:=output.sum())
    s.backward()
    optimizer.step()
    # network.v_mem_init.detach()
print(f"Took {time() - t0:.4f}s")

t0 = time()
for inp in graphed_real_inputs:
    graphed_optimizer.zero_grad()
    output = graphed_network(inp)
    print(s:=output.sum())
    s.backward()
    graphed_optimizer.step()
    # graphed_network.v_mem_init.detach()
print(f"Took {time() - t0:.4f}s")
