# Neural cellular automata to examine behaviours of Self Organized criticality with temporal plasticity
We want to model a cellular automata of the human neurons functional connections. Our cellular automata has the following rules:

1. Neurons can be either in the firing state or the non-firing state
2. Time is discrete.
3. Connections between neurons have a weight that measure the strength of the synaptic connection.
4. Each neuron has a threshold of $\gamma$
5. If a neuron $ A$ is in the not-firing state, we check its neighbors. If there are firing neighbors, we sum the weights of the links directed from such neighbors to neuron $A$. If this sum exceeds a threshold $\gamma$, neuron $A$ changes its state into firing.


To implement Hebbian learning rule with temporal dependence, synaptic weights must be adaptive. For a directed synapsis $A\rightarrow B$, if $A$ fires at time $t$ and $B$ fires at time $t+1$, the weight $w_{AB}$ of the directed connection increases (long-term potentiation). Conversely, if neuron $B$ fires at time $t$ and neuron $A$ fires at time $t+1$, $w_{AB}$ decreases (long-term depression).

This is summed by:
$$
    w_{AB}(t+1) = \begin{cases}
    w_{AB}(t)+l \hspace{1cm} \text{if } A(t)=1, B(t+1)=1 \\
    w_{AB}(t)-f \hspace{0.9cm} A(t+1)=1, B(t)=1 \\
    w_{AB}(t) \hspace{1.6cm} \text{otherwise}
    \end{cases}
$$

1. The Core Network Model: For your 300-neuron model (represented by $N_{v}$), you can use a recurrent weight matrix M to describe interconnections. The steady-state activity vector $v$ of your neurons is determined by: $v= W u + M v$
    - W (Feedforward Matrix): Connections from external inputs (u) to your neurons.
    - M (Recurrent Matrix): Connections between your 300 neurons.
    - v (Output Activity): For your "worm" model, this represents the firing rates of the 300 neurons.
2. Formalizing the Hebbian Rule with "Forgetting": While basic Hebbian rules are unstable and lead to infinite weight growth, the text provides two specific ways to implement your "forgetting" or decay concept.

### Option A: Weight Decay (Simple Forgetting)
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

3. Stability and Constraints: To make your model viable for a 300-neuron scale, you must implement the following constraints mentioned in the text:
Saturation Constraints: Biologically, an excitatory synapse cannot become inhibitory. You should enforce a range for your weights: $ 0 \leq w \leq w_{\max } $
Subtractive Normalization: This is another "forgetting" mechanism where a constant amount is subtracted from all weights to keep their total sum constant. This is highly competitive and helps neurons develop "selectivity" for specific inputs

4. If you implement the recurrent matrix $M$ and the Oja Rule, your simulation loop for each time-step would look like this:


| Step | Concept | Mathematical Formalism | Computational Significance |
| :--- | :--- | :--- | :--- |
| **1** | **Firing Rate Activity** | $v=w\cdot u$  or $v = W\cdot u + M\cdot v$  | Defines the steady-state output ($v$) based on feedforward ($W$) and recurrent ($M$) weight matrices. |
| **2** | **Basic Hebbian Plasticity** | $\tau_{w}\frac{dw}{dt}=vu$  | Models the "correlation" between pre- and postsynaptic activity, where simultaneous activity increases strength. |
| **3** | **Saturation Constraints** | $0 \le w_{b} \le w_{max}$  | Prevents unbounded weight growth and maintains the sign of excitatory/inhibitory synapses. |
| **4** | **Normalization (The "Forgetting" Rule)** | $\tau_{w}\frac{dw}{dt}=vu-av^{2}w$ (**Oja Rule**) or subtractive normalization  | Introduces **competition** and stability by ensuring weights do not grow infinitely. |



