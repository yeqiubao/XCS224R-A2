# Notes

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

---

## Detailed Architecture and Training Mechanics

### 1. The Architecture and Connections

The system is built as a hybrid between **Imitation Learning** and **Reinforcement Learning**.

- **Actor:** The "Worker." It takes an observation and outputs an action. It is initialized by the expert (BC) and then refined by the Critic (RL).
- **Critic:** The "Judge." It learns to predict the expected future reward (Q-value) of an action. It provides the gradient signal that tells the Actor how to improve.
- **Expert Replay Buffer:** The "Curriculum." A fixed set of 20 successful human demonstrations.
- **Agent Replay Buffer:** The "Experience Log." A dynamic storage of every trial and error the robot performs during training.

### 2. How and Why BC is Called

BC is used to solve the **Sparse Reward Problem**. In this environment, the robot only gets a reward (1.0) when the task is fully completed. Without BC, a random robot would never hit the nail and never see a reward.

- **Initialization:** Before RL starts, `agent.bc()` is called for a few thousand steps. This ensures the Actor can already reach for the hammer before the "training wheels" come off.
- **Periodic Regularization:** During the RL phase, `agent.bc()` is often called in every training iteration alongside the RL updates.
- **The "Anchor":** By continuing to call BC, we prevent the **Policy Collapse** seen in your logs. It acts as an anchor that keeps the Actor near the expert's successful path while the Critic is still learning to be accurate.

### 3. The Gradient Update Mechanics

The weights of the Actor are modified by two different forces simultaneously:

- **The BC Gradient:** Calculated via Negative Log-Likelihood (NLL). It pulls the Actor's weights to make its actions look more like the expert's actions.
- **The RL Gradient:** Calculated via the **Reparameterization Trick (`rsample`)**. The Actor asks the Critic: "What is the score for my current action?" The Critic provides a gradient that pulls the Actor's weights toward whatever move yields the highest score.
- **The "Middle Ground":** Mathematically, these updates are applied sequentially. If the Critic gives bad advice (common in early training), the BC gradient "pulls back," forcing the Actor into a compromise between exploration and expert imitation.

### 4. Buffer Evolution During Training

| Buffer | At the Start | During Training | Why? |
|--------|-------------|-----------------|------|
| **Expert Replay Buffer** | Full (20 demos) | **Static** (Never changes) | It represents "perfect" knowledge that shouldn't be corrupted by agent mistakes. |
| **Agent Replay Buffer** | **Empty** | **Dynamic** (Grows every step) | It records the agent's current progress so the Critic can learn from recent successes and failures. |

**The "Seed" Phase:** Before the very first RL update, the robot performs a few hundred steps to "seed" the Agent Replay Buffer so it has a batch of data to sample from. As the buffer reaches its limit (e.g., 100k steps), it acts as a **circular buffer**, deleting the oldest "beginner" mistakes to make room for more advanced experiences.

### 5. Final Implementation Check

To avoid the success rate drop you experienced:

1. **UTD Ratio:** Ensure the Critic is updated more frequently than the Actor (`utd=5`) to keep the "Judge" smarter than the "Worker."
2. **Action Clipping:** Always `clamp(-1, 1)` your actions before passing them to the Critic to match the robot's physical constraints.
3. **Twin-Q Sampling:** Use `random.sample` to ensure you are taking the minimum of two *different* critics to prevent overestimation.


1. BC Loss (Behavior Cloning)Trend: Steady/Low or Not Tracked. Behavior: This loss is typically calculated during the pre-training phase. During the RL fine-tuning phase, if you continue to track it, it should remain low because the policy was already optimized to match the demonstrations. If the policy starts to deviate significantly to find new rewards, this loss might slightly increase as RL takes priority.+32. 
2. Critic LossTrend: Initial Spike, then Gradual Decrease/Stabilization. +2Behavior: At the start of RL, the critic is "surprised" because it is receiving actual environment transitions instead of static demonstrations. As it learns to predict the Bellman targets $y = r_t + \gamma Q(s', a')$, the Mean Squared Error (MSE) should decrease.+3Note: If the success rate is high, the critic loss might fluctuate as the agent discovers new, high-reward states that the critic hasn't fully "valued" yet.
3. Critic Target LossTrend: Delayed and Smoother Version of Critic Loss. +2Behavior: In your code, you don't typically minimize a "Target Loss" directly; rather, the target network provides the ground truth for the main critic. Because the target is updated via Exponential Moving Average (EMA), the values it predicts (the Q-targets) will be more stable and lag behind the main critic's predictions.+2
4. Actor LossTrend: Decreasing (becoming more negative). Behavior: In Actor-Critic, the actor loss is defined as $\mathcal{L}_{\pi_{\theta}} = -\frac{1}{N}\sum Q(s, a)$. As the actor learns to choose actions that the critic gives a higher "score" (higher Q-value), the average Q-value increases. Since the loss is the negative of the Q-value, a successful agent will show an actor loss that trends downward (into negative territory).
