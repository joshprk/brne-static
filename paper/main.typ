#import "@preview/charged-ieee:0.1.3": ieee

#show: ieee.with(
  title: [Incorporating Static Obstacles in Game Theoretic Crowd Navigation],
  abstract: [
  Recent research has shown game theoretic approaches solve the challenging
  problem of optimizing several concurrent objectives in safely navigating
  dense crowds. However, existing frameworks--- such as the Bayesian Recursive
  Nash Equilibrium (BRNE) crowd navigation algorithm---delegate obstacle
  avoidance to a meta-planner, preventing truly unified decision-making and
  requiring external safety guarantees. We present a novel extension to BRNE
  that represents static obstacles as pseudo-agents, where each obstacle
  projects to its closest point to agents' candidate trajectories and applies a
  high-penalty repulsive cost encoding static obstacle collision risk. A
  simulation model demonstrates the practical feasibility of this method. This
  unified approach lays the foundation for crowd navigation that can negotiate
  constraints between static and dynamic obstacles in a single, game-theoretic
  decision process.
  ],
  authors: (
    (
      name: "Joshua Park",
      organization: [Rutgers University],
      location: [New Brunswick, NJ],
      email: "mp1781@scarletmail.rutgers.edu"
    ),
  ),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

#let argmin = math.op("arg min", limits: true)

= Introduction
Crowd navigation is a complicated task which requires the optimization of many
conflicting and non-negotiable objectives in a complex real-time environment.
While maintaining efficient progress towards a goal, producing consistent
pathing, working fast enough to compute just-in-time in a highly dynamic
environment, and reasoning about uncertain external actions, crowd navigation
algorithms must also strictly avoid any collisions with other agents. This
optimization task also appears in several real-world domains other than
pedestrian crowd navigation, from autonomous driving @bal2015 to robotic museum
tour guide robots @burgard1998. Consequently, diverse algorithmic approaches
have emerged--each excelling at specific objectives, but often incurring certain
trade-offs too severe for use in real-world scenarios.

Much of the traditional literature focuses on predict-then-plan architectures
@lecleach2020, where the robot decision-making, or planning, occurs after
trajectories of agents are predicted. The separation between prediction and
planning steps has been known to cause the well-known "freezing robot" problem,
where the robot decides the predicted environment is too unsafe and refuses to
make progress towards its goal @trautman2010. As a result, recent literature has
attempted to merge prediction and planning into a concurrent process. This
effort has been supported by empirical studies of real-world human crowd
movement @feurtey2000, where it is well-documented that collision avoidance
between human agents is often driven by mutual cooperation @bacik2023.

Game theoretic approaches have been shown to perform successful crowd navigation
while combining the prediction and planning steps into one unified process.
However, as Nash

#figure(
  box(
    image("fig1.png", width: 100%),
    clip: true,
    inset: (
      top: -19.5em,
      bottom: -19.5em,
    ),
  ),
  caption: [
    Samples from a distribution of possible trajectories for each agent in a
    crowd navigation problem, where blue and orange represent dynamic agents
    and green represents a static obstacle agent. The original Bayesian
    Recursive Nash Equilibrium (BRNE) algorithm does not guarantee static
    obstacle avoidance without a meta-planner.
  ],
)

equilibrium is computationally intractable for a large number of agents
@daskalakis2006, most early algorithms to solve crowd navigation in a game
theoretic framework use an iterative optimization scheme which converges to
local minima---with no guarantee of a global solution @fridovichkeil2020. In
2024, Sun et al. introduced a Bayesian Recursive Nash Equilibrium (BRNE) solver
for crowd navigation, which provably converges to a global mixed-strategy Nash
equilibrium (MSNE) @sun2024 in real-time. However, the original paper makes no
guarantees about static collision avoidance as that subtask is delegated to the
meta-planner.

Our work investigates the feasibility of encoding static obstacle collision
avoidance into the BRNE algorithm through the representation of static obstacles
as agents. While the original BRNE paper did not see any cases of static
obstacle collision in their testing due to the meta-planner, one should note
that BRNE has no guarantees on static obstacle avoidance @sun2024. Despite
provable guarantees for cooperative pathfinding with other agents, BRNE's
reliance on an external meta-planner leaves static collisions an unaddressed
blind spot. We attempt to integrate that guarantee into the BRNE framework by
generalizing static obstacles as high-penalty repulsive costs treated similarly
to dynamic collision risk in the iterative process. Finally, we validate our
approach in toy-grid simulations, showing that BRNE with static obstacles as
agents guarantees the avoidance of static collisions without an external
meta-planner.

= Related Work

== Predict-Then-Plan
Early works on crowd navigation heavily relied on predict-then-plan paradigm,
where predictions about the environment without the robot were used to produce a
navigation plan that satisfied selected constraints @lecleach2020. The first
public deployment of a robotic crowd navigation algorithm came in 1997, which
utilized value iteration reinforcement learning after mapping a predetermined
area @burgard1998. More recent attempts at using the predict-then-plan paradigm
have involved popular learning techniques such as deep learning techniques. In
Alahi et al., a long short-term memory (LSTM) model was utilized to predict
human trajectories @alahi2016. However, it should be noted that a constant
velocity model has been shown to outperform deep learning methods in trajectory
prediction @scholler2020, hinting that neural networks are poorly suited for the
crowd navigation problem. Scholler et al. particularly notes deep learning
methods are unable to deal with situational priors (such as a car-centric
parking lot in comparison to a pedestrian-centric hallway) and interactions
between agents @scholler2020. There are other predict-then-plan works which
attempt to determine the underlying distribution behing cooperative collision
avoidance, such as the use of inverse reinforcement learning in Henry et al.
@henry2010. These models often either require the use of expensive real-world
data or simulated data, and generally struggle to match learned distributions to
real-world distributions.

Most significantly, predict-then-plan models struggle from the "freezing" robot
problem, where if the environment is predicted to be too complex, the planning
algorithm will decide that any traversal of the crowd is unsafe. It is shown in
Trautman and Krause that the lower bound of an optimal solution to a general
Markov decision process for crowd navigation has an expected cost equivalent to
the cost of a perfect prediction, thus making the freezing robot problem
unavoidable in the predict-then-plan paradigm @trautman2010.

Experimental studies on real-world crowds suggest the freezing robot problem
arises from the predict-then-plan paradigm's underlying assumption that the
robot in charge does not affect the environment around it. Human behavioral
studies suggest human crowds partake in "self-organizing behaviors" where
agent-to-agent interactions show anticipatory behavior in all agents that
explain the crowd navigation ability of humans. In other words, the inability
for other agents to adapt their decision-making distribution when the robot
agent makes a move severely limits a predict-then-plan approach @sun2024.

== Game-Theoretic Local Optimization
Due to the freezing robot problem, many studies now focus on game theoretic
approaches which directly encode cooperation and exploitation into one
optimization problem by solving for a specific type of equilibria. Some studies
encode crowd navigation as a leader-follower game by solving for Stackelberg
equilibria @yoo2012, although the fundamental validity of this is disputed
@sun2024 by other methods as some consider agent equality under Nash equilibrium
to be a more rigorous solution @lecleach2020.

While Stackelberg equilibrium can be computed in polynomial time for crowd
navigation @yoo2012, a general solution to Nash equilibrium is PPAD-complete
@daskalakis2006, and thus incomputable for realtime large-scale crowd navigation
problems using the traditional approach. Game-theoretic approaches have
therefore mostly focused on iterative algorithms which are only guaranteed to
solve for a local minima. For example, Fridovich-Keil et al. recover a local
Nash equilibrium in general-sum differential games by iteratively simulating the
nonlinear trajectories of agents through through a linearization approximation,
approximates a cost, and solves a LQR subproblem to incrementally converge agent
strategies @fridovichkeil2020. Other methods, such as the notable algorithm
ALGAMES, also solve for local minima through other methods such as solving for
KKT conditions to solve for constraints @lecleach2020.

A major concern with local minima solvers is that local minima does not
guarantee efficiency. Failure cases in Monte Carlo simulations often see
severe disagreements between agents due to non-optimal pathing from the agent
stuck in local minima @lecleach2020. In the same study done by Fridovich-Keil et
al., local minima solutions showed dangerous competitive behavior during
simulation @fridovichkeil2020.

== Bayesian Recursive Nash Equilibrium
The Bayesian Recursive Nash Equilibrium (BRNE) solver for crowd navigation is
able to circumvent the PPAD-completeness of solving for global Nash equilibrium
through an equivalent iterative Bayesian update algorithm. Building on the
shortcomings of local-minima solvers, BRNE represents each agent's decision as a
probability distribution over a set of sampeld trajectories, then iteratively
updates the probability mass function of the action distributions to adapt to
agent-to-agent probabilistic collision risks while utilizing KL divergence to
ensure path optimality in real-time. As shown in the appendix of Sun et al.,
BRNE converges to global Nash equilibrium @sun2024.

A notable limitation, however, is that BRNE does not independently encode any
information about static obstacles, which is the focus of this work. Because of
this, BRNE relies entirely on a meta-planner to keep sampled trajectories
obstacle-free. This meta-planner is a separate crowd navigation algorithm
layered on top of BRNE in order to prevent collisions---in this case, a naive
pedestrian-unaware baseline algorithm from Biswas et al. @biswas2021 modified to
support BRNE.

As a result, BRNE alone does not guarantee avoidance of collision with static
obstacles and is in fact completely blind to such environmental hazards.
Consequently, it is unable to incorporate other agents' knowledge of static
obstacles into its decisions nor is it able to make safety decisions by
optimizing distance from static obstacles.

= BRNE-Static
In order to extend BRNE to inherently avoid static obstacles---rather than rely
on a meta-planner---we treat each static obstacle as an additional agent whose
sole purpose is to repel dynamic trajectories. This is possible as the BRNE
objective function still monotonically converges to global Nash equilibrium for
any arbitrary risk function @sun2024:

$
J(p_1, dots.h.c, p_n) &= sum_(i=1)^N sum_(j=i+1)^N integral_S integral_S p(s_i)
p_j(s_j) r(s_i, s_j) dif s_i dif s_j \
  &+ sum_(i=1)^N D(p_i bar.v.double p'_i)
$

Each obstacle is divided into
linear segments and encoded as an agent initialized with a $K$ samples of the
same linear trajectory equivalent to the linear segment. When computing pairwise
risk to optimize the objective function, we differentiate between two different
risk functions in order to support static obstacles.

#h(1em)

1. *Dynamic-Dynamic:* As in the original BRNE algorithm, dynamic-dynamic
  collision risk is determined by an arbitrary collision risk function $r(p_i,
  p_j)$ for dynamic agents $p_i$ and $p_j$.
2. *Dynamic-Static:* For dynamic-static pairs, the dynamic trajectory sample
  involving a point in time with the most minimal orthogonal projection
  distance $argmin_(k,t) d^2_(k,t)$ at a sample $k$ and time $t$ determines the
  magnitude of a smooth inverse logarithmic cost function $c(p_i, p_j)$ for a
  dynamic agent $p_i$ and static agent $p_j$.

#h(1em)

After these pairwise computations are computed, dynamic-static pairs are given a
arbitrary constant multiplier $alpha$ to emphasize the importance of avoiding
collisions in the objective function. It is important to note that the static
collision risk function must be inverse logarithmic to prevent numerical
instability as $min_(k,t) d^2_(k,t) arrow.r 0$. For example, if a dynamic
trajectory sample $k$ passes directly through the static obstacle line, then
there will exist a point $(k, t)$ on that trajectory where $d^2_(k,t) = 0$,
resulting in an infinite cost. An inverse logarithmic collision risk function
for static obstacles solves this as new weights are calculated by passing the
sum of the pairwise risks into an exponential function. For a dynamic agent
$p_i$, the weight update we used for this work is shown below given a set of
dynamic agents $S_d$ and set of static agents $S_s$ where $epsilon = 10^(-6)$:

$
w^*(p_i) &= exp (sum^N_(j in S_d)r(p_i, p_j) + alpha sum^N_(j in S_s)c(p_i, p_j)) \
  &= exp (sum^N_(j in S_d)r(p_i, p_j) + alpha (log (d^2 + epsilon)))
$

Because the static obstacle collision cost function $c(p_i, p_j)$ becomes
arbitrary large for any sample $k$ intersecting the obstacle line, $w_k$ will
immediately be driven to a near-zero probability in the next iteration.
Therefore, over successive updates, only sample trajectories that never cross
the obstacle maintain non-negligible probability mass in the action
distributions, guaranteeing collision-free behavior without additional
meta-planning.

#figure(
  box(
    image("fig2.png", width: 100%),
    clip: true,
    inset: (
      top: -2em,
      bottom: -2em,
    ),
  ),
  caption: [
    A completed two-agent BRNE-Static simulation. It is shown that agents
    successfully avoid obstacles while reaching their goals. The video
    demonstration of this figure is shown at https://drive.google.com/file/d/1eFkwKHbX1NH1ybEkEgHYIWMPUH2xoHCV/view?usp=sharing
  ],
)

= Evaluation

Due to our limited computational resources #footnote[SocNavBench requires a ~90
  GB mesh download, and then a 5-hour compilation stage in order to install the
  benchmark at a minimum] to conduct a rigorous experiment through the original
paper's SocNavBench benchmarks @sun2024, we empirically validate BRNE-Static
through a series of toy-grid experiments to demonstrate its static-obstacle
avoidance without reliance on a meta-planner. Our testing mostly focused on
specific cases such as corridors or dealing with initial positions very close to
the wall, both scenarios with optimal solutions involving high static collision
risk cost. We identified three different types of tests that evaluate the
efficient static obstacle avoidance of BRNE-Static:

#h(1em)

1. *Randomly Cluttered Environments.* These test the general capabilities of
  BRNE-Static by creating game environments containing randomly placed
  obstacles on a 5x5 coordinate grid. Random obstacles were generally 0.25 units
  in length and required to be placed 1.25 units away from each other. These
  tests were performed with three agents.
2. *Simple Corridors.* Two agents were placed in a hallway-like map with two
  walls surrounding them. These agents were expected to pass each other to reach
  their respective goals while avoiding collision with the corridor walls. The
  corridor was 2 units wide and 5 units tall.
3. *Simple Corridors with Corner Turn Ends.* Similarly to the simple corridors
  case, but agents were also required to make a 90-degree turn at the end of the
  corridors. Such turns would generally be handled by the meta-planner, and were
  not hypothesized to be solvable by BRNE-Static due to the action distribution
  samples all being elliptical in shape. The corridor was 2 units wide and 5
  units tall, while the corner exit was 1.5 units wide.

#h(1em)

Each test was randomly generated with the described constraints and performed
100 times each. Each agent action distribution was sampled 200 times and
iterated on 20 times per turn.

Randomly cluttered environments were the simplest challenge for BRNE-Static, as
agents were able to reach their goals with zero collision risks across all
tests. Agents often optimized safety from obstacles, equally utilizing space
between obstacles even when other agents were close. However, it is arguable
that these tests do not reflect real-world realities due to the "seethrough"
properties of the iterative update step. Because BRNE-Static must take into
account all agents to properly converge, it is unable to handle scenarios where
some agents may not know the position of other agents due to obscuring
obstacles. In a real-world scenario, agents would likely be added into the game
modeled by the BRNE algorithm in an iterative step when discovered.

In the simple corridor scenario, 3 out of 100 tests involved one or more
collisions specifically between agents. As the corridor tests had constant
environments, these were the result of poor random seeds due to the necessity of
having to sample the action distribution to ensure the computability of global
Nash equilibrium @sun2024. These collision cases often had both agent take a
near straight-line path, suggesting that the specific random seeds resulted in
trajectory samples that incurred too high of a loss from the static obstacle
collision cost function, thus opting to collide with the other agent instead.

BRNE-Static struggled significantly with the simple corridor with corner turn
case due to the shape of the initial trajectories for dynamic agents. As BRNE
initializes dynamic agents from a uniform Gaussian process centered on the
straight-line path from the start to goal position @sun2024, most trajectories
in the distribution are elliptical shaped. Out of 100 test cases, 24 cases
failed. Out of those 24 failed cases, 20 were due to agent collisions, while 4
were a consequence of obstacle collision. This is the most difficult case for
BRNE-Static, and can potentially make an argument that BRNE-Static may require
augmentation through a meta-planner in order to deal with navigation requiring
sharp turns which are not always in the shape of an ellipsis.

As the baseline meta-planner is only available in the SocNavBench benchmark
@biswas2021, a baseline comparison with the original BRNE is unable to be
performed. However, certain predictions can be made based on the results of the
original paper by Sun et al. BRNE with the SocNavBench meta-planner does not
necessarily optimize for safety distance from static obstacles @sun2024,
potentially resulting in claustrophobic and socially unacceptable pathing in
obstacle-heavy environments. Furthermore, decisions would most likely be much
more interpretable under BRNE-Static as there is only one "mode" of selecting
decisions, whereas a meta-planner can opt to switch between two radically
different modes of decision-making.

= Conclusion
Our extension of BRNE to natively handle static obstacles through pseudo-agent
representations of static obstacles offers key advantages in unifying
decision-making under one game-theoretic model while still maintaining the same
properties that allow BRNE to provably converge to global Nash equilibria.

While our prototype is limited due to dynamic trajectory initialization biased
towards elliptical paths and insufficient computational resources, we believe
that combining static obstacle avoidance and dynamic navigation can show much
more promising results with further research. An unexplored area in this paper is
the use of different collision cost functions, which can heavily influence the
behavior of the outlined framework. Furthermore, the elliptical priors
challenge can be resolved with better prior initialization, as BRNE is still a
novel algorithm with little information of how it behaves outside of uniformly
initialized prior distributions. Despite our work focusing on simpler
environments, we believe BRNE-Static lays a clear foundation for more socially
acceptable static obstacle collision avoidance under a single game theoretic
approach.
