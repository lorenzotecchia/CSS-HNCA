# Neural cellular automata to examine behaviours of Self Organized criticality with temporal plasticity
We want to model a neural cellular automata of the human neurons functional connections. Our cellular automata has the following rules:

1. Time is discrete.
2. Neurons can be either in the firing state or the non-firing state
3. Neurons have 3D coordinates that are randomly generated at the start of the simulation.
4. The neighbors of a neuron $A$ are the neurons that are distant less than a radius $r$ from $A$.
5. A neuron can be connected only to its neighbors.
6. Connections between neurons have a weight that measures the strength of the synaptic connection.
7. The weights have upper and lower bounds $w_max$ and $w_min$, respectively.
8. Each neuron has a threshold of $\gamma$.
9. For each neuron at time $t$, we check its neighbors' state at time $t-1$. If there are firing neighbors, we sum the weights of the links directed from such neighbors to the neuron. If this sum exceeds a threshold $\gamma$, the neuron gets in the firing state. Otherwise, the neuron gets in the non-firing state.
10. (Hebbian learning rule) For a directed synapsis $A\rightarrow B$, if $A$ fires at time $t$ and $B$ fires at time $t+1$, the weight $w_{AB}$ of the directed connection increases (long-term potentiation). Conversely, if neuron $B$ fires at time $t$ and neuron $A$ fires at time $t+1$, $w_{AB}$ decreases (long-term depression).

This is summed by:
$$
w_{AB}(t+1) = \begin{cases}
w_{AB}(t)+l \hspace{1cm} \text{if } A(t)=1, B(t+1)=1 \\
w_{AB}(t)-f \hspace{0.9cm} A(t+1)=1, B(t)=1 \\
w_{AB}(t) \hspace{1.6cm} \text{otherwise}
\end{cases}
$$
where $l$ and $f$ are the learning and forgetting parameter, respectively.

## Mathematical formalization of changing state algorithm.
Let:
- $n$ be the network size be
- $W(t) \in R^{n\times n}$ be the weight matrix of the system at time $t$. The value of coordinate $ij$ is the weight $w_{ij}$ of the connection from node $i$ to node $j$.
- $\vec s(t) \in {0, 1}^n$ be the state vector at time $t$, where

$$
s_i(t) = \begin{cases}
0  \hspace{1cm} \text{if $i$ is not firing} \\
1  \hspace{1cm} \text{if $i$ is firing} \\
\end{cases}
$$
- $\gamma$ be the node threshold. 

Then, the total incoming activity to each neuron is:
$\vec v(t) = W(t-1)^T \cdot \vec s(t-1)$.

The new state is then represented by
$$
s_i(t) =
\begin{cases}
0  \hspace{1cm} \text{if $v_i(t) < \gamma$} \\
1  \hspace{1cm} \text{otherwise} \\
\end{cases}
$$

## Brainstorming:
1. Formalizing the Hebbian Rule with "Forgetting": While basic Hebbian rules are unstable and lead to infinite weight growth, the text provides two specific ways to implement your "forgetting" or decay concept.

### Option A: Weight Decay (Simple Forgetting)\
For supervised learning (where you know the desired output), stability is achieved by adding a simple multiplicative decay term:
$$
\tau_{w} \frac{d w}{d t}=\langle v u\rangle-\alpha w
$$

- $\langle vu \rangle$: The learning term based on the correlation of activity.
− $αw$ : Your "forgetting constant," which forces weights to decay toward zero unless reinforced.

### Option B: The Oja Rule (Competitive Forgetting)
If you are doing unsupervised learning (where the worm learns from its environment), the Oja Rule is a superior formalism. It implements a "forgetting" term that is proportional to the square of the output activity:

$$
\tau_{w} \frac{d w}{d t}=\langle v u\rangle-\alpha v^2 w
$$

Effect: This ensures the "length" of the weight vector stays constant (normalized), preventing weights from growing forever. It naturally introduces competition, where one synapse can only get stronger if others get weaker.

2. Stability and Constraints: To make your model viable for a 300-neuron scale, you must implement the following constraints mentioned in the text:

Saturation Constraints: Biologically, an excitatory synapse cannot become inhibitory. You should enforce a range for your weights: $0  \leq w \leq w_{\max }$

Subtractive Normalization: This is another "forgetting" mechanism where a constant amount is subtracted from all weights to keep their total sum constant. This is highly competitive and helps neurons develop "selectivity" for specific inputs

3. with oja rule your simulation loop for each time-step would look like this:

| Step | Concept | Mathematical Formalism | Computational Significance |
| :--- | :--- | :--- | :--- |
| **1** | **Firing Rate Activity** | $v=W\cdot u$ | Defines the steady-state output ($v$) based on weight matrix $W$. |
| **2** | **Basic Hebbian Plasticity** | $\tau_{w}\frac{dw}{dt}=vu$ | Models the "correlation" between pre- and postsynaptic activity, where simultaneous activity increases strength. |
| **3** | **Saturation Constraints** | $0 \le w_{b} \le w_{max}$ | Prevents unbounded weight growth and maintains the sign of excitatory/inhibitory synapses. |
| **4** | **Normalization (The "Forgetting" Rule)** | $\tau_{w}\frac{dw}{dt}=vu-av^{2}w$ (**Oja Rule**) or subtractive normalization | Introduces **competition** and stability by ensuring weights do not grow infinitely. |