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

[<<Previous](../week-02/README.md) | [Next>>](../README.md)