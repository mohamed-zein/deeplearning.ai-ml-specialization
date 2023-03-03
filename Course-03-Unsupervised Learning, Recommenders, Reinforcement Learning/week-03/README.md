# Reinforcement learning
## Reinforcement learning introduction
### What is Reinforcement Learning?
* How do you get a helicopter to fly itself using reinforcement learning? The task is given the position of the helicopter to decide how to move the control sticks.

$$
\begin{align*}
\text{position of the helicopter} & \longrightarrow \text{how to move the control sticks} \newline
\text{state } s & \longrightarrow \text{action } a
\end{align*}
$$

* Supervised Learning is not a great approach for autonomous helicopter flying.
* A key input to a reinforcement learning is something called the reward or the **reward function** which tells the helicopter when it's doing well and when it's doing poorly.
    * **Positive reward**: helicopter flying well $\longrightarrow +1$.
    * **Negative reward**: helicopter flying poorly $\longrightarrow -1000$.
* Application:
    * Controlling robots.
    * Factory optimization.
    * Financial (stock) trading.
    * Playing games (including video games)
### Mars rover example
![Mars rover example](./images/reinforcement-learning-01.jpg)
* We'll develop reinforcement learning using a simplified example inspired by the Mars rover.
    * In this application, the rover can be in any of six positions, as shown by the six boxes here.
    * The rover, it might start off, say, in disposition into fourth box shown above.
* The position of the Mars rover is called the **state** in reinforcement learning.
* We're going to call these six states, _state 1_, _state 2_, _state 3_, _state 4_, _state 5_, and _state 6_, and so the rover is starting off in _state 4_.
* We would more likely to carry out the science mission ant _state 1_ than at _state 6_, but _state 1_ is further away.
    * The way we will reflect _state 1_ being potentially more valuable is through the reward function. 
    * The reward at _state 1_ is a 100.
    * The reward at _state 6_ is 40.
    * The rewards at all of the other states in-between are zero.
* On each step, the rover gets to choose one of two actions. It can either go to the left or it can go to the right.
* In reinforcement learning, we sometimes state like _state 1_ or _state 6_ this a **terminal state**.
    * What that means is that, after it gets to one of these terminals states, gets a reward at that state, but then nothing happens after that.
* At every step, the robot is in some _state_ $s$, and it gets to choose an action $a$, and it also enjoys some rewards $R(s)$ that it gets from that state. As a result of this action, it to some new state $s'$.

$$
\begin{align*}
(s, a, R(s), s') \newline
(4, \leftarrow, 0, 3)
\end{align*}
$$

> **Note**:  
> The reward $R$ is associated with state $s$ not $s'$

### The Return in reinforcement learning
* How do you know if a particular set of rewards is better or worse than a different set of rewards?
* The concept of a **Return** captures that rewards you can get quicker are maybe more attractive than rewards that take you a long time to get to.
![Return in reinforcement learning](./images/reinforcement-learning-02.jpg)
* The **Return** is defined as the sum of rewards until the _terminal state_ but weighted by one additional factor, which is called the **discount factor** $\gamma$.

$$
\text{Return} = R_{1} + \gamma R_{2} + \gamma^{2} R_{3} + \dots \text{until terminal state}
$$

* What the **discount factor** $\gamma$ does is it has the effect of making the reinforcement learning algorithm a little bit impatient so getting rewards sooner results in a higher value for the total return.
* In many reinforcement learning algorithms, a common choice for the discount factor will be a number pretty close to 1, like 0.9, or 0.99, or even 0.999.

> **Note**:  
> In financial applications, the discount factor also has a very natural interpretation as the **interest rate** or the **time value of money**.

### Making decisions: Policies in reinforcement learning
* There are different ways that you can take actions in the reinforcement learning problem:
    * We could decide to always go for the nearer reward.
    * We could choose actions is to always go for the larger reward.
    * We could always go for smaller reward.
    * We could choose to go left unless you're just one step away from the lesser reward.
* In reinforcement learning, our goal is to come up with a function which is called a **Policy** $\pi$, whose job it is to take as input any state $s$ and map it to some action $a$ that it wants us to take.

$$
\text{state} \quad s \quad \xrightarrow[\pi]{\text{policy}} \quad \text{action} \quad a
$$

> **The goal of reinforcement learning**  
Find a policy $\pi$ that tells you what action $(a=\pi(s))$ to take in every state $(s)$ so as to maximize the retun.

### Markov decision process (MDP)
![Markov decision process](./images/reinforcement-learning-03.jpg)
* This formalism of a reinforcement learning application has a name. It's called a **Markov decision process**.
* The term _Markov_ in the MDP or Markov decision process refers to that the future only depends on the current state and not on anything that might have occurred prior to getting to the current state.
    * In other words, in a Markov decision process, the future depends only on where you are now, not on how you got here.
## State-action value function
### State-action value function definition
* The state action value function is a function typically denoted by $Q(s,a)$ will give a number = the _Return_ if you:
    * start in state $s$.
    * take action $a$ (once).
    * Then behave optimally (take actions that will result in the highest possible return) after that.
![State-action value function](./images/state-action-value-01.jpg)

* The best possible return from state $s$ is $\max\limits_{a} Q(s,a)$.
* The best possible action in state $s$ is the action $a$ that gives $\max\limits_{a} Q(s,a)$.

[<<Previous](../week-02/README.md) | [Next>>](../README.md)