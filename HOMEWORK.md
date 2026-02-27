# XCS224R Assignment 2

**Due:** Sunday, March 8 at 11:59pm PT

---

## Problem 1: Reward Function Impact

### Part 1a

---

### Part 1b

#### Part 1bi

**Successful Manipulation** ($w_0=20, w_1=3, w_2=10, w_3=0.1$, Horizon=2.5)


The agent demonstrates fluid and successful manipulation because high weights on cube positon and orientation force the agent to prioritize the task while a low control penality allows the fingers to move quickly to make necessary changes. The velocity penalty also ensures the sube doesn't fly away due to high speed. Meanwhile, the long 2.5s horizon allows for deep foresight in planning finger trajectories.

---

#### Part 1bii

**Stiff Behavior** ($w_0=20, w_1=3, w_2=10, w_3=1$, Horizon=0.25)

The hand is pretty much still because the increased actuator penalty discourages movement, while the very short 0.25s horizon makes the planner less smart. This combination prevents the agent from planning the complex, multi-stage finger adjustments needed to rotate the cube properly.

---

#### Part 1biii

**Failure and still Behavior** ($w_0=0, w_1=0, w_2=0, w_3=1$, Horizon=2.5)

The hand remains limp or moves away from the cube because all task-related rewards (position, orientation, and velocity) are set to zero. Since the only incentive is to minimize the actuator penalty, the mathematically optimal behavior is for the agent to remain perfectly still, letting the cube fall.

---

## Problem 2: Actor-Critic

### Part 2a: Behavior Cloning [6 points (Coding)]

---

### Part 2b: Actor-Critic Implementation [12 points (Coding)]

#### Update Critic

---

#### Update Actor

---

### Part 2c: Training and Results

#### Part 2ci: Basic Training [6 points (Coding)]

---

#### Part 2cii: Plot Submission [3 points (Written)]

**Plot: eval/episode_success for num_critics=2, utd=1**

To view the plot:
1. Run TensorBoard: `tensorboard --logdir=src/logdir/run_204837_agent.num_critics=2,utd=1/tb`
2. Navigate to the "eval/episode_success" scalar in TensorBoard
3. Take a screenshot or export the plot

**Results:**
- 90% success rate reached at step: _____
- Final success rate: _____
- Total training steps: _____

---

#### Part 2ciii: Extended Training [6 points (Coding)]

---

#### Part 2civ: Comparison and Analysis [5 points (Written)]

---

## Notes and Observations

---

## Submission Checklist

- [ ] Part 1a
- [ ] Part 1bi
- [ ] Part 1bii
- [ ] Part 1biii
- [ ] Part 2a
- [ ] Part 2b
- [ ] Part 2ci
- [ ] Part 2cii
- [ ] Part 2ciii
- [ ] Part 2civ


## How Actor-Critic Works: A Step-by-Step Explanation

### 1. The BC Foundation (Seeding the Success)

Because you start with Behavior Cloning (BC), the robot isn't moving randomly; it follows the 20 expert demonstrations.

The Actor is trained to minimize the negative log-likelihood:

$$\mathcal{L} = -\log \pi_{\theta}(a_{t}|o_{t})$$

This ensures the robot actually reaches the final state where it receives a reward of 1.0. Without this expert data, the Critic would only see 0.0 rewards for millions of steps, and the math would never start working.

### 2. The Critic's Evolution (The "Breadcrumb" Trail)

Once the robot starts hitting that 1.0 reward (thanks to BC), your `update_critic` function uses the Bellman Equation to spread that value:

$$y = r_{t} + \gamma \min(\bar{Q}_{1}, \bar{Q}_{2})$$

Here is how the Critic "evolves" step-by-step:

- **The Goal (Nail Hammered):** The robot gets a reward $r_t = 1.0$. The Critic learns that this state is a 1.0.

- **One Step Before (Hovering over nail):** The reward $r_t$ is 0.0, but the next state has a Q-value of 1.0. The Propagation: Using the discount factor ($\gamma \approx 0.99$), the Critic calculates the value of this step as $0.0 + 0.99(1.0) = \mathbf{0.99}$.

- **Five Steps Before (Reaching for hammer):** This logic continues backward. $0.99$ becomes $0.98$, then $0.97$, and so on.

### 3. The Result: A "Slope" of Values

The Critic evolves from a network that knows nothing into a Value Gradient. It creates a smooth path of increasing numbers that lead to the reward:

| Stage of Task | Reward ($r_t$) | Critic's Score ($Q$) |
|---------------|----------------|---------------------|
| Hand at rest | 0.0 | 0.15 (Long way to go) |
| Grasping the hammer | 0.0 | 0.50 (Halfway there) |
| Hammer near the nail | 0.0 | 0.95 (Almost done) |
| Success (Hit nail) | 1.0 | 1.00 (Perfect Score) |

### 4. The Actor "Climbs" the Slope

Now, the Actor no longer needs the expert demonstrations. In `update_actor`, it samples an action and "asks" the Critic for the score.

The Actor's goal is to maximize the Q-value:

$$\mathcal{L}_{\pi_{\theta}} = -\frac{1}{N}\sum Q_{i}$$

- If the Actor moves the hand away from the hammer, the Critic gives a lower score (e.g., 0.10).
- If the Actor moves toward the hammer, the Critic gives a higher score (e.g., 0.20).

The Actor uses these gradients to "climb" the slope until it consistently reaches the 1.0 reward.
