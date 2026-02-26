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
