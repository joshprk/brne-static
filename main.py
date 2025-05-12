from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import itertools
import random
import math
import copy

DETERMINISTIC_SEED = 42
GOAL_REACHED_CUTOFF = 0.1

class GaussianProcess:
    def __init__(
        self,
        dt: float,
        tsteps: float,
        kernel_a1: float,
        kernel_a2: float
    ) -> None:
        self.kernel_a1 = kernel_a1
        self.kernel_a2 = kernel_a2
        self.Lcov_mat = self.get_Lcov(int(tsteps), dt)

        self.tsteps = tsteps
        self.dt = dt

    def kernel(self, t1: torch.Tensor, t2: torch.Tensor):
        return torch.exp(-self.kernel_a1 * (t1 - t2) ** 2) * self.kernel_a2

    def get_Lcov(self, tsteps: int, dt: float):
        time_list_1 = torch.tensor([0.0, (tsteps - 1) * dt])
        time_list_2 = torch.arange(tsteps) * dt

        mat_11 = torch.vmap(
            self.kernel,
            in_dims=(0, None)
        )(time_list_1, time_list_1)

        mat_12 = torch.vmap(
            self.kernel,
            in_dims=(0, None)
        )(time_list_1, time_list_2)

        mat_22 = torch.vmap(
            self.kernel, in_dims=(0, None)
        )(time_list_2, time_list_2)

        full_mat = mat_22 - mat_12.T @ torch.inverse(mat_11) @ mat_12
        full_mat += torch.eye(tsteps) * 1e-04
        return torch.linalg.cholesky(full_mat)

    def mvn_sampling(self, num_samples: int):
        init_samples = torch.randn(size=(self.Lcov_mat.shape[0], num_samples))
        new_samples = self.Lcov_mat @ init_samples
        return new_samples.T

    def generate_samples(
        self,
        traj_1: torch.Tensor,
        num_samples: int
    ):
        traj_x_samples_1 = self.mvn_sampling(num_samples) + traj_1[:, 0]
        traj_y_samples_1 = self.mvn_sampling(num_samples) + traj_1[:, 1]
        traj_samples_1 = torch.stack([traj_x_samples_1, traj_y_samples_1])
        traj_samples_1 = torch.transpose(traj_samples_1, 0, 1)
        traj_samples_1 = torch.transpose(traj_samples_1, 1, 2)

        return traj_samples_1

@dataclass
class Agent:
    is_static: bool
    init_pos: Tuple[float, float]
    goal_pos: Tuple[float, float]
    _traj: Optional[torch.Tensor] = None
    _weights: Optional[torch.Tensor] = None
    _samples: Optional[torch.Tensor] = None
    _transparent: bool = False

class Updater:
    def __init__(self):
        risk_cov = np.diag(np.random.uniform(low=0.3, high=0.8, size=2)) * 0.5
        risk_cov[0,1] = np.sqrt(np.prod(np.diagonal(risk_cov))) / 2.0
        risk_cov[1,0] = risk_cov[0,1]
        risk_cov = torch.tensor(risk_cov, dtype=torch.float32)
        risk_cov_inv = torch.linalg.inv(risk_cov)
        risk_eta = 1.0 / torch.sqrt((2.0*torch.pi)**2 * torch.linalg.det(risk_cov))

        self._risk_cov = risk_cov
        self._risk_cov_inv = risk_cov_inv
        self._risk_eta = risk_eta

    def risk_fn(self, traj1, traj2):
        d_traj = traj1 - traj2
        inner = -0.5 * torch.sum(d_traj @ self._risk_cov_inv * d_traj, dim=1)
        vals = torch.exp(inner) * 200.0
        return torch.mean(vals)

class Game:
    def __init__(
        self,
        agents = [],
    ):
        self._agents = agents

    def run(
        self,
        dt,
        tsteps,
        num_samples,
        num_iters,
        kernel_a1,
        kernel_a2,
    ):
        gp_pref = GaussianProcess(dt, tsteps, kernel_a1, kernel_a2)
        updater = Updater()

        for agent in self._agents:
            init_pos = agent.init_pos
            goal_pos = agent.goal_pos

            if False and agent.is_static:
                agent._weights = torch.ones(num_samples)
                agent._weights = torch.ones(num_samples) / num_samples
                straight = torch.tensor(
                     np.linspace(init_pos, goal_pos, tsteps),
                     dtype=torch.float32
                 )
                agent._traj = straight
                agent._samples = straight.unsqueeze(0).repeat(num_samples, 1, 1)
            else:
                agent._weights = torch.ones(num_samples)
                agent._traj = torch.tensor(
                    np.linspace(init_pos, goal_pos, tsteps),
                    dtype=torch.float32,
                )

                agent._samples = gp_pref.generate_samples(agent._traj, num_samples)

        risk_tables = {}

        for i, j in itertools.combinations(range(len(self._agents)), 2):
            #if self._agents[i].is_static or self._agents[j].is_static:
            #    continue

            risk_tables[(i, j)] = torch.vmap(
                torch.vmap(updater.risk_fn, in_dims=(None, 0)),
                in_dims=(0,None),
            )(self._agents[i]._samples, self._agents[j]._samples)

        agents = self._agents

        for _ in range(num_iters):
            for i in range(len(agents)):
                if agents[i].is_static:
                    continue

                risks = []

                for j in range(len(agents)):
                    if i == j:
                        continue

                    key = (i, j) if i < j else (j, i)
                    risk = torch.mean(
                        risk_tables[key] * agents[j]._weights,
                        dim=1
                    )

                    risks.append(risk)

                risk_sum = torch.stack(risks).sum(dim=0) / len(risks)
                new_weights = torch.exp(-1.0 * risk_sum)
                new_weights /= torch.mean(new_weights)
                agents[i]._weights = new_weights

        return agents

class AgentAnimator:
    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.T = len(snapshots)
        self.n_agents = len(snapshots[0])

        # set up figure
        self.fig, self.ax = plt.subplots(figsize=(6,6), dpi=100)
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect('equal')
        self.lines = []
        colors = ['C0', 'C1', 'C2', 'C3']

        # draw static obstacle trajectories as dashed lines
        for i, agent in enumerate(snapshots[0]):
            if agent._transparent: continue
            if agent.is_static:
                x0, y0 = agent.init_pos
                x1, y1 = agent.goal_pos
                self.ax.plot(
                    [x0, x1], [y0, y1],
                    linestyle='--', color='k', linewidth=2, label=f'Obstacle {i}'
                )

        # prepare one Line2D per non-static agent
        for i, agent in enumerate(snapshots[0]):
            if agent._transparent: continue
            if not agent.is_static:
                line, = self.ax.plot([], [], '-', lw=2, color=colors[i%len(colors)], label=f'Agent {i}')
                self.lines.append((i, line))

        self.ax.legend(loc='upper right', frameon=False)

    def init(self):
        for _, line in self.lines:
            line.set_data([], [])
        return [ln for _, ln in self.lines]

    def update(self, frame):
        artists = []
        for idx, line in self.lines:
            xs = [snap[idx].init_pos[0] for snap in self.snapshots[:frame+1]]
            ys = [snap[idx].init_pos[1] for snap in self.snapshots[:frame+1]]
            line.set_data(xs, ys)
            artists.append(line)
        return artists

    def animate(self):
        self.ani = animation.FuncAnimation(
            self.fig, self.update, frames=self.T,
            init_func=self.init, blit=True, interval=100
        )
        return self.ani

    def save(self, filename='agents.mp4'):
        if not hasattr(self, 'ani'):
            self.animate()
        self.ani.save(filename, writer='ffmpeg', fps=10)

def plot(result):
    _, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")

    for i in range(len(result)):
        agent = result[i]

        if agent._transparent: continue

        traj = agent._traj.cpu().numpy()
        samples = agent._samples.cpu().numpy()
        weights = agent._weights.cpu().numpy()

        ax.plot(traj[0,0], traj[0,1], marker="o", markersize=15, color="C" + str(i))

        for traj_i, w_i in zip(samples, weights):
            ax.plot(
                traj_i[:,0],
                traj_i[:,1],
                linestyle="-",
                linewidth=3,
                color="C" + str(i),
                alpha=np.minimum(0.05 * w_i, 1.0) if not agent.is_static else 1.0,
            )

    _, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")

    for i in range(len(result)):
        agent = result[i]

        if agent._transparent: continue

        traj = torch.mean(agent._samples * agent._weights.view(-1, 1, 1), dim=0)

        ax.plot(traj[0,0], traj[0,1], marker="o", markersize=15, color="C" + str(i))
        ax.plot(traj[:,0], traj[:,1], linestyle="-", linewidth=3, color="C" + str(i), alpha=1.0)

    plt.show()
    plt.close()

def goals_reached(agents):
    for agent in agents:
        if not agent.is_static:
            dist = math.dist(agent.init_pos, agent.goal_pos)

            if not (dist <= GOAL_REACHED_CUTOFF):
                return False

    return True

def take_step(agents):
    for i in range(len(agents)):
        agent = agents[i]

        if agent.is_static:
            continue

        traj = torch.mean(agent._samples * agent._weights.view(-1, 1, 1), dim=0)
        agent.init_pos = traj[1]

    return agents

def main():
    agents = [
        Agent(False, (3.0, 0.0), (-1.5, 0.0)),
        Agent(True, (0.0, 1.5), (0.0, -1.5)),
        Agent(False, (0.0, 2.0), (0.0, -2.0), _transparent=True),
        #Agent(False, (0.0, -2.0), (0.0, 2.0), _transparent=True),
    ]

    snapshots = []

    try:
        while not goals_reached(agents):
            game = Game(agents)

            result = game.run(
                dt=0.3,
                tsteps=20,
                num_samples=200,
                num_iters=20,
                kernel_a1=0.03,
                kernel_a2=20.0
            )

            agents = take_step(agents)
            #plot(agents)
            snapshots.append(copy.deepcopy(agents))
    except KeyboardInterrupt:
        pass
    finally:
        animator = AgentAnimator(snapshots)
        animator.save()

if __name__ == "__main__":
    torch.random.manual_seed(DETERMINISTIC_SEED)
    np.random.seed(DETERMINISTIC_SEED)

    main()
