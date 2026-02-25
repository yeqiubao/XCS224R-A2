# XCS224R Module 2: Offline Reinforcement Learning - Key Concepts and Terminology

## Table of Contents
1. [Recap: Online RL Methods](#recap-online-rl-methods)
2. [Offline Reinforcement Learning](#offline-reinforcement-learning)
3. [Why Offline RL?](#why-offline-rl)
4. [Challenges with Off-Policy Methods](#challenges-with-off-policy-methods)
5. [Offline RL Approaches](#offline-rl-approaches)
6. [Reward Learning](#reward-learning)
7. [Learning from Human Preferences](#learning-from-human-preferences)

---

## Recap: Online RL Methods

### Policy Gradient Methods

**Policy Gradient Objective:**
$$\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(\mathbf{a}_{i,t} | \mathbf{s}_{i,t}) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_{i,t'}, \mathbf{a}_{i,t'}) \right) - b \right)$$

**Key Terminology:**
- $\pi_{\theta}(\mathbf{a} | \mathbf{s})$: Policy parameterized by $\theta$
- $\mathbf{s}_{i,t}$: State at time $t$ in trajectory $i$
- $\mathbf{a}_{i,t}$: Action at time $t$ in trajectory $i$
- $r(\mathbf{s}, \mathbf{a})$: Reward function
- $b$: Baseline (variance reduction)

**Intuition:** Do more of the above-average actions, less of the below-average actions.

### Actor-Critic Methods

**Actor-Critic Policy Gradient:**
$$\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(\mathbf{a}_{i,t} | \mathbf{s}_{i,t}) A^{\pi}(\mathbf{s}_{i,t}, \mathbf{a}_{i,t})$$

**Key Terminology:**
- $A^{\pi}(\mathbf{s}, \mathbf{a})$: Advantage function = $Q^{\pi}(\mathbf{s}, \mathbf{a}) - V^{\pi}(\mathbf{s})$
- $Q^{\pi}(\mathbf{s}, \mathbf{a})$: Action-value function (expected return from state-action pair)
- $V^{\pi}(\mathbf{s})$: State-value function (expected return from state)

**Intuition:** Estimate what is good and bad, then do more of the good stuff.

### Value Function Estimation

**1. Monte Carlo Estimation of $V^{\pi}$:**
$$\min_{\phi} \sum_{\mathbf{s}_t \sim \mathcal{D}} \left\| \hat{V}_{\phi}^{\pi_{\theta}}(\mathbf{s}_t) - \sum_{t' = t}^{T} r(\mathbf{s}_t, \mathbf{a}_t) \right\|^2$$

**2. Bootstrapped/TD Estimation of $V^{\pi}$:**
$$\min_{\phi} \sum_{(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim \mathcal{D}} \left\| \hat{V}_{\phi}^{\pi_{\theta}}(\mathbf{s}) - \left(r(\mathbf{s}, \mathbf{a}) + \gamma \hat{V}_{\phi}^{\pi_{\theta}}(\mathbf{s}')\right) \right\|^2$$

**3. Bootstrapped/TD Estimation of $Q^{\pi}$:**
$$\min_{\phi} \sum_{(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim \mathcal{D}} \left\| \hat{Q}_{\phi}^{\pi_{\theta}}(\mathbf{s}, \mathbf{a}) - \left(r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_{\mathbf{a}' \sim \pi_{\theta}(\cdot | \mathbf{s}')} [\hat{Q}_{\phi}^{\pi_{\theta}}(\mathbf{s}', \mathbf{a}')]\right) \right\|^2$$

**Key Terminology:**
- $\hat{V}_{\phi}^{\pi_{\theta}}(\mathbf{s})$: Estimated state-value function (parameterized by $\phi$)
- $\hat{Q}_{\phi}^{\pi_{\theta}}(\mathbf{s}, \mathbf{a})$: Estimated action-value function (parameterized by $\phi$)
- $\gamma$: Discount factor
- $\mathcal{D}$: Dataset of transitions
- $\mathbb{E}_{\mathbf{a}' \sim \pi_{\theta}(\cdot | \mathbf{s}')}$: Expectation over actions sampled from policy

### Full Off-Policy Actor-Critic Algorithm

**Algorithm Steps:**
1. Take action $\mathbf{a} \sim \pi_{\theta}(\mathbf{a}|\mathbf{s})$, get $(\mathbf{s}, \mathbf{a}, \mathbf{s}', r)$, store in $\mathcal{R}$
2. Sample a batch $\{\mathbf{s}_i, \mathbf{a}_i, r_i, \mathbf{s}_i'\}$ from buffer $\mathcal{R}$
3. Update $\hat{Q}_{\phi}^{\pi}$ using targets:
   $$y_{i} = r_{i} + \gamma \hat{Q}_{\phi}^{\pi}(\mathbf{s}_{i}^{\prime}, \mathbf{a}_{i}^{\prime})$$
   where $\mathbf{a}_i' \sim \pi_{\theta}(\cdot | \mathbf{s}_i')$
4. Policy gradient update:
   $$\nabla_{\theta}J(\theta)\approx \frac{1}{N}\sum_{i}\nabla_{\theta}\log \pi_{\theta}(\mathbf{a}_{i}^{\pi}|\mathbf{s}_{i})\hat{Q}^{\pi}(\mathbf{s}_{i},\mathbf{a}_{i}^{\pi})$$
   where $\mathbf{a}_i^\pi \sim \pi_\theta (\mathbf{a}|\mathbf{s}_i)$
5. Update parameters: $\theta \gets \theta + \alpha \nabla_{\theta} J(\theta)$

**Key Terminology:**
- $\mathcal{R}$: Replay buffer
- $\alpha$: Learning rate

---

## Offline Reinforcement Learning

### Definition

**Offline RL Process:**
- Given a **static dataset** $\mathcal{D}$
- Train policy on the provided dataset (no online data collection)

**Key Difference from Online RL:**
- **Online RL:** Collect data → Update policy → Collect more data (iterative)
- **Offline RL:** Given static dataset → Train policy (no new data collection)

### Why Offline RL?

**Motivations:**
1. **Leverage existing datasets:** Use data collected by people or existing systems
2. **Safety concerns:** Online policy collection may be risky or unsafe
3. **Data reuse:** Reuse previously collected data rather than recollecting
   - Previous experiments
   - Previous projects
   - Data from other robots/institutions

**Note:** A blend of offline then online RL is also possible!

### Formal Setup

**Offline Dataset:**
$$\mathcal{D} = \{(\mathbf{s}, \mathbf{a}, \mathbf{s}', r)\}$$

Sampled from some unknown **behavior policy** $\pi_{\beta}$:

$$\mathbf{s} \sim p_{\pi_{\beta}}(\cdot)$$
$$\mathbf{a} \sim \pi_{\beta}(\cdot \mid \mathbf{s})$$
$$\mathbf{s}' \sim p(\cdot \mid \mathbf{s}, \mathbf{a})$$
$$r = r(\mathbf{s}, \mathbf{a})$$

**Key Terminology:**
- $\pi_{\beta}$: Behavior policy (the policy that collected the data)
- $p_{\pi_{\beta}}(\cdot)$: State distribution under behavior policy
- Note: $\pi_{\beta}$ may be a mixture of policies

**Objective:**
$$\max_{\theta} \mathbb{E}_{p_{\theta}(\tau)} \left[ \sum_{t} r(\mathbf{s}_t, \mathbf{a}_t) \right]$$

**Key Challenge:** Distribution shift - expectation is under learned policy $\pi_{\theta}$, but data comes from $\pi_{\beta}$.

### Data Sources

Where does the data come from?
- Human collected data
- Data from a hand-designed system/controller
- Data from previous RL run(s)
- A mixture of sources

---

## Challenges with Off-Policy Methods

### The Problem

**Off-Policy Critic Objective:**
$$\min_{\phi} \sum_{(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim \mathcal{D}} \left\|\hat{Q}_{\phi}^{\pi_{\theta}}(\mathbf{s}, \mathbf{a}) - \left(r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_{\mathbf{a}' \sim \pi_{\theta}(\cdot | \mathbf{s}')}[\hat{Q}_{\phi}^{\pi_{\theta}}(\mathbf{s}', \mathbf{a}')]\right)\right\|^2$$

**Policy Update:**
$$\nabla_{\theta}J(\theta)\approx \frac{1}{N}\sum_{i}\nabla_{\theta}\log \pi_{\theta}(\mathbf{a}_{i}^{\pi}|\mathbf{s}_{i})\hat{Q}_{\phi}^{\pi_{\theta}}(\mathbf{s}_{i},\mathbf{a}_{i}^{\pi})$$
where $\mathbf{a}_i^\pi \sim \pi_\theta (\mathbf{a}|\mathbf{s}_i)$

### What Goes Wrong?

**Key Issue:** When evaluating $Q$ on actions $\mathbf{a}' \sim \pi_{\theta}(\cdot | \mathbf{s}')$ that are **out-of-distribution** (not in the dataset):

1. **Extrapolation error:** $Q$-function may be inaccurate for actions not seen in the dataset
2. **Distribution shift:** Policy $\pi_{\theta}$ may select actions very different from behavior policy $\pi_{\beta}$
3. **Overestimation:** $Q$-function may be overestimated for out-of-distribution actions, leading to poor policy updates

**Result:** Policy may exploit errors in $Q$-function estimates, leading to poor performance.

---

## Offline RL Approaches

### Two Main Approaches

1. **Implicit Policy Constraint Methods**
   - Constrain the learned policy to stay close to the behavior policy
   - Examples: Behavior Cloning, BCQ (Batch Constrained Q-learning)

2. **Conservative Methods**
   - Penalize or downweight out-of-distribution actions in $Q$-function
   - Examples: CQL (Conservative Q-Learning), TD3+BC

### Key Learning Goals

- Understand the key challenges arising in offline reinforcement learning
- Understand two approaches for offline RL and why they work
- Understand how offline RL can improve over imitation learning

---

## Reward Learning

### Motivation

**Key Insight:** Rewards can't be taken for granted!

In many real-world scenarios:
- Reward functions are difficult to specify manually
- We may have examples of desired behavior or outcomes
- We may have human preferences but not explicit rewards

### Where Do Rewards Come From?

1. **Manual specification:** Hand-designed reward functions
2. **Learning from example goals:** Learn reward from goal examples
3. **Learning from demonstrations:** Learn reward from expert demonstrations
4. **Learning from human preferences:** Learn reward from pairwise comparisons

---

## Learning from Human Preferences

### Setup

**Human Feedback:**
- Human says trajectory $\tau_{w}$ is better than $\tau_{l}$
- Notation: $\tau_{w} > \tau_{l}$
- Note: $\tau$ could be a full or partial roll-out

**Objective:**
We want a reward $r_{\theta}$ such that:
$$\sum_{(s,a)\in \tau_w}r_\theta (s,a) > \sum_{(s,a)\in \tau_l}r_\theta (s,a)$$

**Key Insight:** Humans are classifying which trajectory is better. Reward should be discriminative as well.

### Reward Learning Algorithm

**Probability Model:**
Define $\sigma(r_{\theta}(\tau_a) - r_{\theta}(\tau_b))$ as the estimated probability that $\tau_{a} > \tau_{b}$.

**Objective:**
$$\max_{\theta} \mathbb{E}_{\tau_w, \tau_l}[\log \sigma(r_\theta(\tau_w) - r_\theta(\tau_l))]$$

**Key Terminology:**
- $\sigma(\cdot)$: Sigmoid function
- $r_{\theta}(\tau)$: Reward for trajectory $\tau$ under reward model parameterized by $\theta$

### Complete Algorithm

1. **Data Collection:**
   - Given dataset $\{\tau_i\}$, sample batches of $k$ trajectories
   - Ask humans to rank trajectories
   - (For LLMs, these $k$ trajectories all have the same prompt)

2. **Reward Computation:**
   - Compute $r_{\theta}(\tau_1), \ldots, r_{\theta}(\tau_k)$ under current reward model $r_{\theta}$

3. **Gradient Update:**
   - For all $\binom{k}{2}$ pairs per batch, compute:
     $$\nabla_{\theta} \mathbb{E}_{\tau_w, \tau_l} \left[ \log \sigma(r_{\theta}(\tau_w) - r_{\theta}(\tau_l)) \right]$$
     where $\tau_w > \tau_l$

4. **Parameter Update:**
   - Update $\theta$ using computed gradient

**Note:** This can be done in the loop of online RL.

### Applications

**1. Learning Rewards in the Loop of Online RL:**
- Uses human preference queries during training
- Example: Learning rewards for driving with different factors:
  - $w_{1}$ for Road Boundary
  - $w_{2}$ for Staying within Lanes
  - $w_{3}$ for Keeping Speed
  - $w_{4}$ for Heading
  - $w_{5}$ for Collision Avoidance

**2. Learning Rewards for LLMs:**

**Three-Stage Process:**
1. **Large-scale pre-training:** Next token prediction, mixed quality
2. **Supervised fine-tuning:** On higher-quality (prompt, response) pairs
3. **RL from human feedback (RLHF):**
   - **3a. Gather preference data:** Sample replies, ask human which is better
   - **3b. Train reward model:** Train $r(x, y)$ that judges how good response $y$ is for prompt $x$
   - **3c. Reinforcement learning:** Finetune model to predict $y$ for prompt $x$ with high reward (e.g., using PPO)

**Reinforcement Learning with AI Feedback (RLAIF):**
- Ask another language model "which of these responses is less harmful?"
- **Key insight:** Critique is easier than generation!

### Summary of Reward Learning

**Learning rewards from goals/demos:**
- ✅ Practical framework for task specification
- ⚠️ Adversarial training can be unstable (though variety of regularization tricks from GAN literature)
- ❌ Requires examples of desired behavior or outcomes

**Learning rewards from human preferences:**
- ✅ Pairwise preferences easy to provide (doesn't require example goals, demos!)
- ✅ Has been deployed at scale!
- ❌ May require supervision in the loop of RL (usually requires more human time)

---

## Additional Concepts

### Generative Adversarial Networks (GANs) Connection

**How GANs work:**
1. Train classifier to discriminate between real data and generated data
2. Train generator to generate data that the classifier thinks is real

**At convergence:** Generator will match data distribution $p(x)$

**Connection to Reward Learning:**
- Pre-trained classifiers can be **exploited** when optimized against
- Solution: Update the classifier *during RL*, using policy data as negatives
- Can learn goal classifier with success examples, full reward with demos

### Unsupervised RL

**Can RL agents propose their own goals?**

**Example: Two-Player Game:**
- **Goal-setter:** Proposes goals
- **Goal-reacher:** Tries to achieve goals

**Self-Play Episode:** No supervision -- internal reward only

**Target Task Episode:** Supervision from external reward

**Key Insight:** Agents can learn to set their own goals through self-play.

---

## Key Terminology Summary

### Policy and Value Functions
- **Policy:** $\pi_{\theta}(\mathbf{a} | \mathbf{s})$ - Probability distribution over actions given state
- **State-value function:** $V^{\pi}(\mathbf{s})$ - Expected return from state under policy $\pi$
- **Action-value function:** $Q^{\pi}(\mathbf{s}, \mathbf{a})$ - Expected return from state-action pair under policy $\pi$
- **Advantage function:** $A^{\pi}(\mathbf{s}, \mathbf{a}) = Q^{\pi}(\mathbf{s}, \mathbf{a}) - V^{\pi}(\mathbf{s})$

### Offline RL Specific
- **Behavior policy:** $\pi_{\beta}$ - The policy that collected the offline dataset
- **Distribution shift:** Mismatch between data distribution (from $\pi_{\beta}$) and policy distribution (from $\pi_{\theta}$)
- **Out-of-distribution actions:** Actions not seen in the dataset
- **Extrapolation error:** Errors in $Q$-function estimates for out-of-distribution actions

### Reward Learning
- **Reward model:** $r_{\theta}(\mathbf{s}, \mathbf{a})$ or $r_{\theta}(\tau)$ - Learned reward function
- **Preference:** $\tau_w > \tau_l$ - Human preference indicating trajectory $\tau_w$ is better than $\tau_l$
- **RLHF:** Reinforcement Learning from Human Feedback
- **RLAIF:** Reinforcement Learning with AI Feedback

### General RL
- **Replay buffer:** $\mathcal{R}$ - Storage for transitions $(\mathbf{s}, \mathbf{a}, \mathbf{s}', r)$
- **Discount factor:** $\gamma$ - Factor for future rewards
- **Baseline:** $b$ - Used for variance reduction in policy gradients
- **Learning rate:** $\alpha$ - Step size for parameter updates

---

## Mathematical Notations

### Common Symbols
- $\mathbf{s}$: State
- $\mathbf{a}$: Action
- $\mathbf{s}'$: Next state
- $r$: Reward
- $\tau$: Trajectory (sequence of states and actions)
- $\theta$: Policy parameters
- $\phi$: Value function parameters
- $\mathcal{D}$: Dataset
- $\mathbb{E}$: Expectation
- $\sim$: "Sampled from" or "distributed as"
- $\sigma(\cdot)$: Sigmoid function
- $\log$: Natural logarithm
- $\max$: Maximum
- $\min$: Minimum
- $\sum$: Summation
- $\prod$: Product
- $\binom{n}{k}$: Binomial coefficient (number of ways to choose $k$ from $n$)

### Operators
- $\nabla_{\theta}$: Gradient with respect to $\theta$
- $\approx$: Approximately equal
- $\| \cdot \|$: Norm (usually L2 norm)
- $p(\cdot)$: Probability distribution
- $\pi(\cdot | \mathbf{s})$: Conditional probability distribution (policy)

---

*This summary is based on XCS224R Module 2: Offline Reinforcement Learning*
