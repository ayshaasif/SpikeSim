import numpy as np

def pe_tile_count(in_ch_list, out_ch_list, out_dim_list, k, xbar, pe_per_tile):
    num_layer = len(out_ch_list)
    pe_list = []
    for i in range(num_layer):
        num_pe = np.ceil(in_ch_list[i] / xbar) * np.ceil(out_ch_list[i] / xbar)
        pe_list.append(num_pe)
    
    print(f"PEs required per layer: {pe_list}")
    
    # --- FIXED: Removed the CNN-specific strict sorting check constraint ---
    num_pe = sum(pe_list)
    print(f'Total No. of PEs: {num_pe}')

    if (num_pe % pe_per_tile != 0):
        num_tiles = (num_pe // pe_per_tile) + 1
    else:
        num_tiles = num_pe / pe_per_tile

    print(f'Total No. of Tiles Allocated: {num_tiles}')
    return int(num_tiles)

def compute_energy(in_ch_list, in_dim_list, out_ch_list, out_dim_list, xbar_size, k, n_tiles, device_list, time_steps):
    # Supporting components constants
    Tile_buff, Temp_Buff, Sub, ADC = 397, 0.2, 1.15E-6, 2.03084
    Htree, MUX, mem_fetch, neuron = 19.64 * 8, 0.094245, 4.64, 1.274 * 4.0

    energy_layerwise = []
    tot_energy = 0
    
    for i in range(len(out_ch_list)):
        device = device_list[i]
        xbar_ar = 1.76423 if device == 'rram' else 671.089 
        
        PE_ar = k * k * xbar_ar + (xbar_size / 8) * (ADC + MUX)
        PE_cycle_energy = Htree + mem_fetch + neuron + xbar_size / 8 * PE_ar + (xbar_size / 8) * 16 * Sub + (xbar_size / 8) * Temp_Buff + Tile_buff
        
        Total_PE_cycle = np.ceil(out_ch_list[i] / xbar_size) * np.ceil(in_ch_list[i] / xbar_size) * (out_dim_list[i] * out_dim_list[i])
        layer_energy = Total_PE_cycle * PE_cycle_energy * time_steps
        
        tot_energy += layer_energy
        energy_layerwise.append(layer_energy)

    print(f'Total Hybrid Model Energy: {tot_energy:,.2f} pJ')
    return energy_layerwise

def compute_area(in_ch_list, in_dim_list, out_ch_list, out_dim_list, xbar_size, k, pe_per_tile, n_tiles, device_list):
    Tile_buff, Temp_Buff, Sub, ADC = 0.7391 * 64 * 128, 484.643999, 13411.41498, 693.633
    Htree, MUX = 216830 * 2, 45.9

    total_pe_area = 0
    for i in range(len(out_ch_list)):
        device = device_list[i]
        xbar_ar = 26.2144 if device == 'rram' else 671.089 
        
        num_pe = np.ceil(in_ch_list[i] / xbar_size) * np.ceil(out_ch_list[i] / xbar_size)
        layer_pe_ar = k * k * xbar_ar + (xbar_size / 8) * (ADC + MUX)
        total_pe_area += num_pe * layer_pe_ar

    Tile_ar_ov_base = (xbar_size / 8) * Sub + Temp_Buff + Tile_buff + Htree
    total_compute_ar_ov = (Tile_ar_ov_base * n_tiles) + total_pe_area + np.sum(np.array(in_ch_list) * np.array(in_dim_list) * np.array(in_dim_list)) * 0.7391 * 22
    neuron_mem = np.sum(np.array(out_ch_list[0:5]) * np.array(out_dim_list[0:5]) * np.array(out_dim_list[0:5])) * 0.7391 * 22
    total_ar = total_compute_ar_ov + neuron_mem

    print(f'Total Hybrid Model Compute Area: {total_ar:,.2f} µm^2')
    return total_ar

def latency_gen(in_ch_list, out_ch_list, out_dim_list, k, xbar, pe_per_tile, PE_cycle, time_steps):
    num_layer = len(out_ch_list)
    checkpoints_2 = []
    
    # --- FIXED: Removed the CNN spatial ascending sorting pipeline assertion check ---
    for i in range(num_layer):
        checkpoints_2.append(int(out_dim_list[i] * out_dim_list[i] * out_ch_list[i]) * time_steps)
    
    halts = np.cumsum(checkpoints_2) 
    print(f'Final Model Latency: {(halts[-1]) * PE_cycle:,.2f} ns')

# ==========================================
# NEW CONFIGURATION: Spiking Transformer Layer Block
# ==========================================

# Hardware Constraint Parameters
xbar_size = 64
pe_per_tile = 8
time_steps = 5
clk_freq = 250  # MHz
PE_cycle = 26 * (1000 / clk_freq)

# --- Transformer Software Dimensions ---
# Sentence Hyperparameters: "Read the book" padded out or structured sequences
L = 64   # Sequence Length (Number of Tokens)
H = 256  # Embedding Size / Hidden Dimension

# Formulating structural layer mapping lists:
# Layer 0: W_q Projection (Linear layer)
# Layer 1: W_k Projection (Linear layer)
# Layer 2: W_v Projection (Linear layer)
# Layer 3: THE DUMMY ATTENTION LAYER (Q x K^T) -> (L x H) * (H * L) = (L x L)
# Layer 4: W_o Post-Attention Projection (Linear layer)
# Layer 5: MLP Feed-Forward Layer 1 Expansion
# Layer 6: MLP Feed-Forward Layer 2 Shrinkage

in_ch_list   = [H, H, H,  H, H, H,   H*4] 
out_ch_list  = [H, H, H,  L, H, H*4, H]   
in_dim_list  = [1, 1, 1,  L, 1, 1,   1]   # Linear projections run on vectors (dim=1)
out_dim_list = [1, 1, 1,  L, 1, 1,   1]   # The dynamic attention expands over L matrix dims

# Attention step uses kernel size 1 (direct matrix dot-product interaction)
# Projections represent flat Linear evaluations, effectively making k=1
kernel_size = 1 

# --- The Hybrid Core Component Map ---
# Projections are loaded once into RRAM. Attention relies on high-endurance, volatile SRAM.
device_list = ['rram', 'rram', 'rram', 'sram', 'rram', 'rram', 'rram']

# --- Execute Updated Pipeline Profile ---
print("--- Spiking Transformer Hardware Profiling Results ---")
n_tiles = pe_tile_count(in_ch_list, out_ch_list, out_dim_list, kernel_size, xbar_size, pe_per_tile)

if n_tiles is not None:
    compute_area(in_ch_list, in_dim_list, out_ch_list, out_dim_list, xbar_size, kernel_size, pe_per_tile, n_tiles, device_list)
    compute_energy(in_ch_list, in_dim_list, out_ch_list, out_dim_list, xbar_size, kernel_size, n_tiles, device_list, time_steps)
    latency_gen(in_ch_list, out_ch_list, out_dim_list, kernel_size, xbar_size, pe_per_tile, PE_cycle, time_steps)
    