
fundamental concept - the "Memory Wall" paradox of Compute-In-Memory (CIM) hardware design.

To clear up the confusion, we need to look at what "dynamic weights" means in this context and how a CIM array handles a matrix multiplication when **neither operand is fixed**.

---

### The Fundamental Rule of CIM (The Anchor)

In any standard Compute-In-Memory crossbar array, **one of the two matrix operands must be physically stationary** inside the memory cells (as programmed conductances), while the other operand is fed into the rows as a **streaming voltage activation signal**.

* In a **CNN**: The weights are static. They are loaded into the RRAM cells *once* and stay there forever. The image features stream in as input spikes.
* In a **Transformer ($Q \times K^T$)**: Both $Q$ and $K$ are dynamic activations that change with every word. They are both "streaming inputs."

So how can a CIM array multiply them if CIM requires one operand to act as the stationary "weight"?

---

### How CIM Makes an Activation Look Like a Weight

To calculate $\text{Attention} = Q \times K^T$, the hardware compiler treats **one of the dynamic activation streams as a temporary, short-term stationary weight matrix.** During inference, the step-by-step logic inside the chip works like this:

#### Step 1: Generating the Vectors

The token sequence flows through the static weight projections ($W_Q$ and $W_K$) which are permanently sitting on RRAM tiles. The chip outputs two streaming pools of spikes: the **Queries ($Q$)** and the **Keys ($K$)**.

#### Step 2: The "Fake Weight" Loading Phase

Instead of streaming both $Q$ and $K$ through wires simultaneously and trying to smash them together mid-air, the compiler pauses.

It takes the newly calculated Key activations ($K^T$) and **writes them directly into a blank, volatile CIM memory array** (typically high-speed SRAM or eDRAM bit-cells designed for quick rewriting). For the duration of this specific sentence, **the Key activations are now frozen inside the memory cells, acting as the "weights."**

#### Step 3: Streaming the Query

Now that the Key matrix is locked inside the grid, the Query matrix ($Q$) is shot into the array rows as streaming input voltage spikes.

$$\text{Input Spikes (Query)} \times \text{Stored Memory State (Key)} = \text{Attention Score Output}$$

#### Step 4: Wipe and Repeat

The moment the attention calculation for this block is finished, those memory cells are unlatched, wiped, and prepared to receive the Key activations of the next layer or the next token window.

---

### What you are simulating in SpikeSim

When a hardware paper or a simulator like SpikeSim talks about "dynamic weights" at inference, they are referring to the **runtime overhead of rewriting those volatile CIM arrays.**

| Attribute | Static CNN Mapping | Dynamic Transformer Attention ($Q \times K^T$) |
| --- | --- | --- |
| **Memory Type** | Permanent (RRAM / Non-Volatile) | Temporary (SRAM / eDRAM / Volatile) |
| **Write Phase Overhead** | **Zero** at inference (Written once at the factory). | **High** at inference (Must overwrite memory cells for every token/sequence layer). |
| **Energy Impact** | Purely dynamic reading/firing energy. | High write energy because of constant bit-cell charging and flipping. |

### Summary of the Concept

The weights aren't dynamic because the *model* is changing its mind; they are dynamic because **the software "activations" of the Key matrix are physically drafted to serve as the hardware "weights" inside the CIM array for a split-second.** This is exactly why your current code needs that "dummy layer" modification. The simulator needs to know the dimensions of $Q$ and $K$ so it can accurately calculate how much clock latency and energy the chip wastes constantly writing and rewriting those temporary key tokens into the local Tile blocks!



## CNN vs Transformer in CIM

In a traditional Convolutional Neural Network, **100% of the layers are built using static, permanent weights.** Every single layer in a CNN—whether it is a convolutional layer or a fully connected layer—is made of a fixed matrix of numbers (the filter kernels) that were learned during training. Once the model is trained, those numbers never change.

Because of this, a hardware compiler can map **every single operation** of a CNN onto permanent, non-volatile RRAM crossbars at the factory. There are no dynamic matrix-by-matrix multiplications like the $Q \times K^T$ step found in Transformers.

---

### The Big Difference: Why Transformers are Unique

To visualize why your simulator handles a CNN perfectly but needs a "tweak" for a Transformer, look at what the inputs and weights actually represent during execution:

#### 1. Inside a CNN Layer

* **The Inputs (Dynamic Spikes):** The changing pixel values of the input image. They flow through the wires as electrical pulses.
* **The Matrix (Static Conductances):** The $3 \times 3$ kernel weights. They are permanently frozen inside the RRAM crossbar cells.
* **The Result:** The chip multiplies the moving spikes by the frozen weights. **(Your simulator handles this perfectly.)**

#### 2. Inside the Transformer Attention Layer

* **The Inputs (Dynamic Spikes):** The Query ($Q$) spikes generated by the current token sentence.
* **The Matrix (Dynamic Spikes!):** The Key ($K^T$) spikes generated by the current token sentence.
* **The Result:** The chip is forced to multiply **moving spikes by moving spikes**.

---

### Why your CNN simulator needs that "Dummy Layer" trick

Because a CNN has zero "spike-by-spike" matrix layers, your simulator's code assumes that **every layer passed into it has a fixed set of weights stored in a crossbar.**

When you move to a Spiking Transformer, the $Q \times K^T$ attention calculation still takes up physical space on the chip (temporary SRAM buffers) and consumes a massive amount of dynamic energy routing those spikes together.

By adding a **dummy layer** to your Python lists with an input channel size of $L$ (Sequence Length) and an output channel size of $L$, you are essentially "tricking" your CNN simulator. You are telling it: *"Hey, pretend there is an extra layer here with these dimensions so you can calculate the massive power and hardware overhead of smashing those two token streams together."*


---

## Step 1: The Core Transformer "Story" (The Software)

Imagine a standard Transformer is trying to process a short sequence of text: **"Read the book"**.

* **Sequence Length ($L = 3$):** There are 3 words/tokens in our sequence.
* **Embedding Size ($H = 256$):** Each word is converted into a vector of 256 feature numbers.

In software (like PyTorch), this input matrix $X$ has a shape of `[3, 256]`. To calculate attention, the Transformer needs three vectors for each word: a **Query** (what the word is looking for), a **Key** (what the word contains), and a **Value** (the actual content).

We get these by multiplying our input $X$ against three trained, static weight matrices ($W_Q, W_K, W_V$):


$$Q = X \cdot W_Q$$

$$K = X \cdot W_K$$

$$V = X \cdot W_V$$

Once we have these, the core **Attention Step** happens:


$$\text{Attention Score} = Q \times K^T$$

This multiplies $Q$ (`[3, 256]`) by $K^T$ (`[256, 3]`) to yield a final `[3, 3]` matrix. This output is a map showing how much "Read", "the", and "book" relate to each other.

---

## Step 2: The Hardware Wall (Silicon Reality)

When you look at a Compute-In-Memory (CIM) neuromorphic chip, you aren't dealing with abstract matrices anymore; you are dealing with **crossbar arrays** of a fixed physical size (like $64 \times 64$).

A crossbar operates strictly on **Ohm's Law ($I = V \cdot G$)**:

* **$V$ (Voltage Input):** Fed into the horizontal rows. It must be **dynamic** (changing every clock cycle).
* **$G$ (Conductance Memory):** Programmed into the physical intersections of the grid. It must be **stationary/frozen** while the voltage passes through.
* **$I$ (Current Output):** Read out at the bottom of the columns as the result of the matrix multiplication.

Because your embedding dimension ($H = 256$) is larger than a single crossbar grid ($64$), SpikeSim has to spatially chop your math up into a grid of $\lceil 256 \div 64 \rceil = 4$ crossbar rows.

---

## Step 3: Mapping the Transformer to the Chip (The SpikeSim Use Case)

Here is where the magic happens. A Spiking Transformer requires a **hybrid chip** because its layers have completely different physical properties.

### 1. The Projection Layers ($W_Q, W_K, W_V$) use RRAM

The weights $W_Q, W_K,$ and $W_V$ are learned during training and never change during inference.

* **The Hardware:** We program them into permanent, non-volatile **RRAM cells** once.
* **The Cycle System:** Because we don't have enough wires to feed the whole sentence at once, the chip uses clock cycles. At **Cycle 1**, the vector for "Read" enters the rows as voltages, hits the RRAM weights, and outputs $\vec{q}_1$ and $\vec{k}_1$. At **Cycle 2**, "the" streams in. At **Cycle 3**, "book" streams in.

### 2. The Attention Layer ($Q \times K^T$) MUST use SRAM

Now we have a massive dilemma for the $\text{Attention} = Q \times K^T$ calculation. Both $Q$ and $K$ are dynamic activations changing with every single sentence. But Ohm's law requires one of them to act as a frozen conductance ($G$) inside the memory cells!

* **The Hardware Trick:** The hardware controller takes the newly calculated Key activations ($K^T$) and forces them into a high-speed, volatile **SRAM crossbar array**. For a split second, $K^T$ is **frozen** into the SRAM cells to serve as the conductance ($G$).
* **The Cycle System:** While $K^T$ sits frozen in the SRAM cells, the Query vectors ($Q$) are released into the rows cycle-by-cycle as voltages ($V$):
* *Cycle 1:* $\vec{q}_1$ ("Read") streams across the frozen $K^T$ matrix $\rightarrow$ Outputs Row 1 of the attention map.
* *Cycle 2:* $\vec{q}_2$ ("the") streams across the exact same frozen $K^T$ matrix $\rightarrow$ Outputs Row 2.
* *Cycle 3:* $\vec{q}_3$ ("book") streams across $\rightarrow$ Outputs Row 3.


* **The Wipe:** The moment Cycle 3 finishes, the SRAM cells are instantly wiped to prepare for the next sentence.





In this In-Memory Computing (IMC) architecture, **cycles represent time.** Specifically, they represent discrete ticks of the hardware's internal clock (the digital control bus).

Because the hardware grid is fixed in size ($64 \times 64$), you cannot dump an entire Transformer sequence or an entire SNN time-window onto the chip all at once. The hardware must break the data apart and process it piece-by-piece, **using clock cycles to serialize the operations.**

There are actually **two distinct types of cycles** working together inside your SpikeSim framework. Understanding their roles explains exactly how the chip computes and why your latency numbers look the way they do.

---

### 1. Macro-Cycles: SNN Temporal Steps ($T$)

* **What it represents:** The programmatic timeline of your Spiking Neural Network (e.g., `time_steps = 5`).
* **The Role:** SNNs process information over a temporal window. If an image or a token sequence is passed to the network, the input spikes are fed into the chip over 5 distinct, consecutive intervals. The Leaky Integrate-and-Fire (LIF) neurons inside the Processing Elements (PEs) use these macro-cycles to accumulate current on their internal membrane capacitors, deciding whether to fire an output spike at step $t_1$, $t_2$, up to $t_5$.

---

### 2. Micro-Cycles: Hardware Computation Ticks (`PE_cycle`)

* **What it represents:** The raw hardware processing time required for a single physical layer to complete one operational pass (e.g., your code's definition: `PE_cycle = 26` hardware clock cycles).
* **The Role:** A physical crossbar cannot execute instantly. When an input vector of voltage spikes hits the rows, a chain reaction of analog and digital events must occur sequentially. Those 26 clock cycles inside the PE are split into highly specific roles:

```
[ 26 Hardware Clock Cycles per PE Pass ]
 ├── Phase 1 (1-2 cycles): Input Latching -> Row voltages stabilize.
 ├── Phase 2 (10-12 cycles): Analog Settling -> Current flows through cells.
 ├── Phase 3 (8-10 cycles): ADC Conversion -> Analog column current converted to digital bits.
 └── Phase 4 (2-3 cycles): Neuron Update -> LIF neuron steps and fires output spikes.

```

---

### The Role of Cycles in the Transformer Attention Step ($Q \times K^T$)

To see how these two cycle concepts collide during your Spiking Transformer simulation, let's look at the exact timeline role they play when computing your **Dummy Attention Layer** (64 tokens):

1. **Cycle 0 (Setup):** The Key matrix ($K^T$) is loaded and latched into the SRAM crossbar cells. This is a one-time hardware overhead cycle.
2. **Cycles 1 through 64 (The Token Stream):** Because you have 64 tokens ($L=64$), the hardware must feed the Query ($Q$) matrix into the rows **one token per micro-cycle block**.
* *Cycle 1:* Query vector for Token 1 enters $\rightarrow$ Row 1 of the Attention matrix is calculated.
* *Cycle 2:* Query vector for Token 2 enters $\rightarrow$ Row 2 of the Attention matrix is calculated.
* This continues sequentially until **Cycle 64** completes the matrix.


3. **The SNN Multiplier ($T=5$):** Because this entire sequence evaluation must happen across your SNN temporal timeline, this 64-cycle loop is repeated **5 separate times** to let the spikes integrate over the network's time-steps.

### Why Your Simulator Cares About Cycles

Your `latency_gen(...)` function uses these cycle definitions to compute real-world execution speeds. It multiplies the total accumulated hardware cycles by your clock period ($\frac{1000}{250 \text{ MHz}} = 4\text{ ns}$).

If your compiler calculates that a Spiking Transformer block requires 5,000 clock cycles to completely stream, map, and process the attention matrices, your simulation output will report a hard physical latency of **20,000 ns** ($5000 \times 4\text{ ns}$) for that inference pass.


Let’s trace a real, concrete example to see exactly how a Spiking Transformer handles a sentence, why it gets chopped up, and how the math distributes over **hardware clock cycles**.

Imagine you are deploying a Spiking Transformer to process this simple sentence:

> **"Read the book"**

---

### Step 1: The Software Setup (The Data Dimensions)

Before the sentence touches the silicon chip, your software turns these words into numbers (tokens and embeddings):

1. **Sequence Length ($L = 3$):** There are 3 words/tokens in our sentence ("Read", "the", "book").
2. **Embedding Size ($H = 256$):** Each word is turned into a list of 256 feature numbers.
3. **Time Steps ($T = 5$):** Because this is a *Spiking* network, every single number is converted into a stream of binary spikes (1s and 0s) over 5 consecutive time intervals.

In your Python code, your input data matrix looks like a giant grid of `[3, 256]`.

---

### Step 2: The Physical Hardware Constraints

Your neuromorphic chip has a physical layout that looks like a clean grid of crossbar arrays:

* **The Array Grid Size ($X = 64$):** Each individual hardware crossbar grid is exactly $64 \times 64$. It has 64 horizontal input wires and 64 vertical output lines.
* **The Input Pin Limit:** The entry port for this specific layer on the chip only has **256 physical wires** leading into it.

---

### Step 3: Why the Chip Cannot Take the Full Data At Once

Look at the conflict between your software data and the physical silicon:

* Your total sentence data is $3 \text{ words} \times 256 \text{ features} = 768$ individual values per SNN time-step.
* Your chip's input port only has **256 physical wires**.

If the chip tried to swallow all 3 words at the exact same fraction of a microsecond, it would need $3 \times 256 = 768$ physical input wires leading into that block. The chip simply doesn't have the space for that many parallel copper lines.

Therefore, the hardware controller **serializes** the data—it forces the words to line up in a queue and enter the chip **one word at a time over consecutive clock cycles**.

---

### Step 4: Tracking the Inference Timeline (Cycle by Cycle)

Let’s watch the chip process the **Query Projection Layer** ($W_Q$) during a single SNN time-step. Because the embedding size is 256, the chip wires 4 crossbars together horizontally ($\lceil 256 / 64 \rceil = 4$) to create a 256-wire wide input bus.

Here is the exact hardware schedule:

#### Cycle 1: Processing the word "Read"

* **The Input:** The chip takes the 256 spike channels for the first word, **"Read"**, and floods them onto the 256 input wires.
* **The Math:** The electrical pulses travel through the crossbars, multiply by the stored weights via Ohm's Law, and accumulate at the bottom columns.
* **The Output:** The column ADCs convert the analog current into digital spike signals. Out comes a new 256-channel vector: **Query Vector 1**.

#### Cycle 2: Processing the word "the"

* **The Input:** The 256 input wires are cleared. The chip now pumps the 256 spike channels for the second word, **"the"**, onto the *exact same 256 wires*.
* **The Math:** Physics repeats the multiplication process inside the exact same crossbars.
* **The Output:** Out comes a new 256-channel vector: **Query Vector 2**.

#### Cycle 3: Processing the word "book"

* **The Input:** The wires are cleared again. The 256 spike channels for the final word, **"book"**, fill the input lines.
* **The Math:** The final multiplication runs through the grid.
* **The Output:** Out comes the final 256-channel vector: **Query Vector 3**.

---

### Summary of the Engineering Reality

By spreading the calculation out over **3 clock cycles**:

1. The hardware successfully processed a full `[3, 256]` sentence matrix.
2. It only needed **256 physical input wires** instead of 768 wires, saving massive amounts of silicon space.
3. It reused the exact same arithmetic crossbar multipliers and expensive Analog-to-Digital Converters (ADCs) 3 times over, maximizing the efficiency of the physical hardware footprint.

This is exactly why your simulator code calculates latency by multiplying your layer steps by `PE_cycle` loops. It is measuring the physical time the network spends waiting for these token queues to stream through those hard-wired silicon boundaries!

