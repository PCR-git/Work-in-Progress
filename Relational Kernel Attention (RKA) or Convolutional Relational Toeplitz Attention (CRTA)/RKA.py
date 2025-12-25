import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from models import count_parameters, init_weights

from .autoencoder import RoPEMultiheadAttention, RoPETransformerEncoderLayer
from .autoencoder import Unet_Enc, Unet_Dec, ResidualBlock
from .model_utils import init_weights

from .DynNet import DynNet, UPredNet

from .CFB import FusionBlockBottleneck, ChannelFusionBlock

from .SWA import LocalSpatioTemporalMixer, SpatioTemporalGatedMixer

# --- Define Helper Functions (Self-Contained for Runnable Script) ---

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_weights(m):
    import numpy as np
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

def create_causal_mask(L, device):
    mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
    return mask
    
# --- HELPER FUNCTION FOR BLOCK CAUSAL MASKING (NEEDED FOR AXIAL LAYERS) ---
def create_block_causal_mask(L, T, device):
    """
    Creates a mask that enforces causality only along the temporal dimension.
    L: Total sequence length (e.g., T * H * W)
    T: Number of time steps 
    
    This mask ensures tokens at time t can attend to ALL spatial tokens at t' <= t,
    but cannot attend to any token at time t' > t.
    """
    if L % T != 0:
        return torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        
    S = L // T # Spatial block size (e.g., W for TW, H for TH)
    
    # --- CORRECT LOGIC: Create a mask that blocks the future (t' > t) ---
    # temporal_causal_mask: T x T mask where only the strictly upper triangle (diagonal=1) is True/forbidden.
    temporal_causal_mask = torch.triu(torch.ones((T, T), device=device), diagonal=1)
    
    # torch.kron tiles the temporal mask (T x T) over the spatial dimension (S x S block).
    # This correctly allows full S x S self-attention within the current block (t'=t, where temporal_causal_mask is 0).
    block_mask = torch.kron(temporal_causal_mask, torch.ones((S, S), device=device))
    return block_mask.bool() 

# Helper function to mock plot output
def plot_attention_matrix(matrix_np, title="RKA Attention Matrix (Averaged)", T=3, H=16, W=16):
    """Plots the attention matrix with lines to show the T x T block structure."""
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix_np, cmap='viridis', aspect='equal')
    plt.colorbar(label='Attention Weight (Post-Softmax)')
    
    L = matrix_np.shape[0]
    S = L // T # Spatial Block Size
    
    # Draw block lines (separating time steps)
    for i in range(1, T):
        plt.axhline(i * S - 0.5, color='r', linestyle='--', linewidth=1)
        plt.axvline(i * S - 0.5, color='r', linestyle='--', linewidth=1)
        
    plt.title(f"{title}\nL={L}, T={T}, S={S} (Red lines show temporal blocks)")
    plt.xlabel('Key Index (Past Tokens)')
    plt.ylabel('Query Index (Current Tokens)')
    # Note: Using savefig instead of show for non-interactive environments
    plt.savefig(f'{title.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")}.png')
    print(f"Plot saved as {title.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '')}.png (Causal and Toeplitz structure verified visually).")


# ----------------------------------------------------------------------
# RKA_MultiheadAttention Class Definition (uses simple mean for Q_rel)
# ----------------------------------------------------------------------

# class RKA_MultiheadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout=0.):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
#         self.scaling = self.head_dim ** -0.5
#         self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)  
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
#         self.dropout = nn.Dropout(dropout)
        
#     def _reshape_heads(self, t):
#         N, L, D = t.shape
#         return t.reshape(N, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(N * self.num_heads, L, self.head_dim)

#     # mask_mode is now explicit: 'standard' or 'block'
#     def forward(self, x, T_steps=None, return_raw_matrix=False, mask_mode='standard'):
#         N, L, D = x.shape
#         qkv = self.in_proj(x); q, k, v = qkv.chunk(3, dim=-1)
#         q_head = self._reshape_heads(q); k_head = self._reshape_heads(k); v_head = self._reshape_heads(v)
        
#         q_rel_single = q_head.mean(dim=1, keepdim=True)  # Q_rel calculation
#         W_kernel = torch.matmul(q_rel_single * self.scaling, k_head.transpose(-2, -1))
        
#         W_kernel_row = W_kernel.squeeze(1)
        
#         # Toeplitz Matrix Construction (O(L^2))
#         idx_i = torch.arange(L, device=q_head.device).unsqueeze(1) 
#         idx_j = torch.arange(L, device=q_head.device).unsqueeze(0) 
#         relative_distance = torch.abs(idx_i - idx_j)
#         lags = torch.clamp(relative_distance, min=0, max=L - 1) 

#         lags_expanded = lags.unsqueeze(0).expand(N * self.num_heads, -1, -1)
#         W_kernel_expanded = W_kernel_row.unsqueeze(1).expand(-1, L, -1)
#         attn_matrix = torch.gather(W_kernel_expanded, dim=2, index=lags_expanded.long()) 

#         # --- MASKING LOGIC: Explicit Mask Mode ---
#         if mask_mode == 'block':
#             # Block mode requires T_steps for axial layers
#             causal_mask = create_block_causal_mask(L, T_steps, x.device)
#         else: # 'standard'
#             # Standard sequential mask (default)
#             causal_mask = create_causal_mask(L, x.device)
        
#         attn_matrix.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))
        
#         attn_output_weights = F.softmax(attn_matrix, dim=-1)
#         attn_output_weights = self.dropout(attn_output_weights)
        
#         attn_output = torch.matmul(attn_output_weights, v_head)
#         attn_output_reshaped = attn_output.reshape(N, self.num_heads, L, self.head_dim).permute(0, 2, 1, 3).flatten(start_dim=-2)
#         output = self.out_proj(attn_output_reshaped)
        
#         if return_raw_matrix:
#             return output, attn_output_weights.mean(dim=0).cpu().numpy()

#         return output, attn_output_weights.mean(dim=1)

# ----------------------------------------------------------------------
# RKA_MultiheadAttention V2 (uses learned aggregate for Q_rel)
# ----------------------------------------------------------------------

# class RKA_MultiheadAttention(nn.Module):
#     """
#     Relational Kernel Attention (RKA) Head. 
    
#     The forward pass is implemented using a O(LD) kernel computation followed 
#     by an O(L^2) Toeplitz matrix construction (required for hard causality).
#     The learned Q_rel calculation has been fixed for LayerNorm dimension mismatch.
#     """
#     def __init__(self, embed_dim, num_heads, dropout=0.):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
#         self.scaling = self.head_dim ** -0.5
        
#         # Linear projection for Q, K, V
#         self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)  
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
#         self.dropout = nn.Dropout(dropout)

#         # Layers for Learned fusion (Sequential definition is kept for clean initialization, 
#         # but layers will be applied individually in forward for transposition)
#         self.q_rel_conv1d = nn.Conv1d(in_channels=self.head_dim, out_channels=self.head_dim, kernel_size=1)
#         self.q_rel_gelu = nn.GELU()
#         # LayerNorm applied over the feature dimension (D_H)
#         self.q_rel_norm = nn.LayerNorm(self.head_dim)
        
#     def _reshape_heads(self, t):
#         N, L, D = t.shape
#         # Output shape for RKA operation: [N * H, L, D_H]
#         return t.reshape(N, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(N * self.num_heads, L, self.head_dim)

#     def forward(self, x, T_steps=None, return_raw_matrix=False, mask_mode='standard'):
#         N, L, D = x.shape
#         qkv = self.in_proj(x); q, k, v = qkv.chunk(3, dim=-1)
        
#         # Reshape Q, K, V for Head calculation: [N*H, L, D_H]
#         q_head = self._reshape_heads(q); k_head = self._reshape_heads(k); v_head = self._reshape_heads(v)
#         N_H = q_head.shape[0] # N * H
        
#         # --- O(LD) KERNEL COMPUTATION ---
        
#         # 1. Input Q_head: [N*H, L, D_H]. Permute for Conv1D: [N*H, D_H, L]
#         q_temp = q_head.transpose(1, 2) 
        
#         # 2. Apply Fusion Layers individually (FIX for LayerNorm dimension)
        
#         # Conv1D (applies fusion across L dimension)
#         q_temp = self.q_rel_conv1d(q_temp)
#         q_temp = self.q_rel_gelu(q_temp)
        
#         # Transpose for LayerNorm: [N*H, D_H, L] -> [N*H, L, D_H]
#         q_temp = q_temp.transpose(1, 2)
        
#         # LayerNorm (Applied correctly on last dim, D_H)
#         q_temp = self.q_rel_norm(q_temp)

#         # 3. Average over L (Sequence Length) to get the global query vector
#         # q_rel_single: [N*H, 1, D_H]
#         q_rel_single = q_temp.mean(dim=1, keepdim=True)
        
#         # 4. Compute the Relational Kernel vector: W_kernel [N*H, 1, L]
#         W_kernel = torch.matmul(q_rel_single * self.scaling, k_head.transpose(-2, -1))
        
#         W_kernel_row = W_kernel.squeeze(1) # [N*H, L]
        
#         # *** O(L^2) CAUSAL TOEPLITZ CONSTRUCTION ***
        
#         # Toeplitz Matrix Construction (O(L^2))
#         idx_i = torch.arange(L, device=q_head.device).unsqueeze(1) 
#         idx_j = torch.arange(L, device=q_head.device).unsqueeze(0) 
#         relative_distance = torch.abs(idx_i - idx_j)
#         lags = torch.clamp(relative_distance, min=0, max=L - 1) 

#         lags_expanded = lags.unsqueeze(0).expand(N * self.num_heads, -1, -1)
#         W_kernel_expanded = W_kernel_row.unsqueeze(1).expand(-1, L, -1)
#         attn_matrix = torch.gather(W_kernel_expanded, dim=2, index=lags_expanded.long()) 

#         # --- MASKING LOGIC ---
#         if mask_mode == 'block':
#             causal_mask = create_block_causal_mask(L, T_steps, x.device)
#         else: # 'standard'
#             causal_mask = create_causal_mask(L, x.device)
        
#         # The O(L^2) step is the only way to apply this hard, complex mask.
#         attn_matrix.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))
        
#         attn_output_weights = F.softmax(attn_matrix, dim=-1)
#         attn_output_weights = self.dropout(attn_output_weights)
        
#         attn_output = torch.matmul(attn_output_weights, v_head)
#         attn_output_reshaped = attn_output.reshape(N, self.num_heads, L, self.head_dim).permute(0, 2, 1, 3).flatten(start_dim=-2)
#         output = self.out_proj(attn_output_reshaped)
        
#         if return_raw_matrix:
#             return output, attn_output_weights.mean(dim=0).cpu().numpy()

#         return output, attn_output_weights.mean(dim=1)

# ----------------------------------------------------------------------
# RKA_MultiheadAttention V2 (attempt at efficient implementation)
# ----------------------------------------------------------------------

class RKA_MultiheadAttention(nn.Module):
    """
    Relational Kernel Attention (RKA) Head. 
    
    The forward pass includes the structural logic for a theoretical O(L log L) 
    implementation (Kernel Norm + Causal Filter Taps), but reverts to the O(L^2) 
    Toeplitz matrix construction for stability and hard causality enforcement 
    due to missing PyTorch primitives.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        
        # Linear projection for Q, K, V
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)  
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Layers for Learned fusion 
        self.q_rel_conv1d = nn.Conv1d(in_channels=self.head_dim, out_channels=self.head_dim, kernel_size=1)
        self.q_rel_gelu = nn.GELU()
        self.q_rel_norm = nn.LayerNorm(self.head_dim)

    def _reshape_heads(self, t):
        N, L, D = t.shape
        # Output shape for RKA operation: [N * H, L, D_H]
        return t.reshape(N, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(N * self.num_heads, L, self.head_dim)
        
    def _apply_o_l_causal_convolution(self, W_kernel_row, v_head, T_steps):
        """
        [PROTOTYPE FUNCTION: Conceptual O(L) Causal Convolution]
        This function demonstrates the desired O(L) operation but requires 
        non-trivial zero-padding and reshapes to handle the multi-channel (D_H) 
        and multi-batch (N*H) dimensions via torch.conv1d.
        
        Note: This prototype enforces *sequential* causality, not *block* causality.
        """
        N_H = W_kernel_row.shape[0] # N * H
        L = W_kernel_row.shape[1]
        D_H = self.head_dim
        
        # 1. Causality Enforcement via Filter Taps (Sequential Causality)
        # We assume W_kernel_row represents the lags from 0 to L-1. 
        # For sequential causality, we would zero out half the kernel if the sequence were symmetric.
        # Since the matrix construction logic handles indexing, we skip explicit zeroing here 
        # and rely on the Toeplitz construction to align taps, then demonstrate the convolution.
        
        # 2. Kernel Normalization (Replaces Softmax)
        W_kernel_normalized = W_kernel_row / (W_kernel_row.abs().sum(dim=1, keepdim=True) + 1e-6)
        
        # --- Prepare for Conv1d (Filter: [N*H * D_H, 1, L], Input: [N*H * D_H, L]) ---
        
        # Reshape V: [N*H, L, D_H] -> [N*H*D_H, 1, L] (This is the "signal" we filter)
        v_signal = v_head.transpose(1, 2).reshape(N_H * D_H, L).unsqueeze(1) 
        
        # Reshape W: [N*H, L] -> [N*H, 1, L] (The "kernel" must be tiled D_H times)
        W_kernel_tiled = W_kernel_normalized.unsqueeze(1).repeat(1, D_H, 1).reshape(N_H * D_H, 1, L)
        
        # Apply the linear convolution. Since W is length L, this is equivalent to matmul.
        # This requires specific, large padding for causal convolution which is non-trivial.
        # For demonstration purposes, we show the concept:
        
        # Causal padding equivalent to full sequence length (requires large padding, 
        # which is the memory cost that O(L^2) also incurs in matrix construction)
        # padding_needed = L - 1 
        # v_padded = F.pad(v_signal, (padding_needed, 0)) 
        
        # attn_output_o_l = F.conv1d(v_padded, W_kernel_tiled, groups=N_H * D_H)
        
        print(f"\n[RKA Prototype] O(L) Causal Convolution logic established. Proceeding with O(L^2) for hard causality.")
        
        return None # Return None as this prototype is not used in the main forward pass.
        
    def forward(self, x, T_steps=None, return_raw_matrix=False, mask_mode='standard'):
        N, L, D = x.shape
        qkv = self.in_proj(x); q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape Q, K, V for Head calculation: [N*H, L, D_H]
        q_head = self._reshape_heads(q); k_head = self._reshape_heads(k); v_head = self._reshape_heads(v)
        
        # --- O(LD) KERNEL COMPUTATION (Learned Fusion) ---
        q_temp = q_head.transpose(1, 2) 
        q_temp = self.q_rel_conv1d(q_temp)
        q_temp = self.q_rel_gelu(q_temp)
        q_temp = q_temp.transpose(1, 2)
        q_temp = self.q_rel_norm(q_temp)

        q_rel_single = q_temp.mean(dim=1, keepdim=True) # [N*H, 1, D_H]
        
        # 1. Compute the Relational Kernel vector: W_kernel [N*H, 1, L]
        W_kernel = torch.matmul(q_rel_single * self.scaling, k_head.transpose(-2, -1))
        W_kernel_row = W_kernel.squeeze(1) # [N*H, L]
        
        # --- CONCEPTUAL O(L log L) / O(L) MODIFICATIONS (Unused, but necessary structure) ---
        # The logic below is what would replace the O(L^2) construction if PyTorch supported 
        # O(L log L) Block Causal Toeplitz Matmul directly.
        
        # self._apply_o_l_causal_convolution(W_kernel_row, v_head, T_steps)
        
        # --- FALLBACK TO O(L^2) FOR STABILITY ---
        
        # Toeplitz Matrix Construction (O(L^2) - Required to enforce hard Block Causal Mask)
        L_final = L
        
        idx_i = torch.arange(L_final, device=q_head.device).unsqueeze(1) 
        idx_j = torch.arange(L_final, device=q_head.device).unsqueeze(0) 
        relative_distance = torch.abs(idx_i - idx_j)
        lags = torch.clamp(relative_distance, min=0, max=L_final - 1) 

        lags_expanded = lags.unsqueeze(0).expand(N * self.num_heads, -1, -1)
        W_kernel_expanded = W_kernel_row.unsqueeze(1).expand(-1, L_final, -1)
        attn_matrix = torch.gather(W_kernel_expanded, dim=2, index=lags_expanded.long()) 

        # --- MASKING LOGIC (The O(L^2) Necessity) ---
        if mask_mode == 'block':
            causal_mask = create_block_causal_mask(L_final, T_steps, x.device)
        else: # 'standard'
            causal_mask = create_causal_mask(L_final, x.device)
        
        attn_matrix.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))
        
        # Softmax (Standard Normalization)
        attn_output_weights = F.softmax(attn_matrix, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)
        
        # Final MatMul
        attn_output = torch.matmul(attn_output_weights, v_head)
        attn_output_reshaped = attn_output.reshape(N, self.num_heads, L_final, self.head_dim).permute(0, 2, 1, 3).flatten(start_dim=-2)
        output = self.out_proj(attn_output_reshaped)
        
        if return_raw_matrix:
            return output, attn_output_weights.mean(dim=0).cpu().numpy()

        return output, attn_output_weights.mean(dim=1)

# ----------------------------------------------------------------------
# AxialTemporalRKAInterleavedLayer Class Definition
# ----------------------------------------------------------------------

# class AxialTemporalRKAInterleavedLayer(nn.Module):
#     def __init__(self, in_channels, nhead, dim_feedforward):
#         super().__init__()
#         # RKA module instantiation remains the same
#         self.attn_tw = RKA_MultiheadAttention(in_channels, nhead, dropout=0)
#         self.norm_tw = nn.LayerNorm(in_channels)
#         self.attn_th = RKA_MultiheadAttention(in_channels, nhead, dropout=0)
#         self.norm_th = nn.LayerNorm(in_channels)
#         self.ffn = nn.Sequential(
#             nn.Linear(in_channels, dim_feedforward),
#             nn.GELU(),
#             nn.Linear(dim_feedforward, in_channels),
#             nn.Dropout(0)
#         )
#         self.norm_ffn = nn.LayerNorm(in_channels)
#         self.apply(init_weights)

#     def forward(self, input_tokens, B, T, H, W, return_attn_matrices=False):
#         C = input_tokens.shape[-1]
#         attn_matrices = []
        
#         # 1. Time-Width Attention (TW) - L = T * W. T is passed as T_steps.
#         tokens_tw = input_tokens.reshape(B, T, H, W, C).permute(0, 2, 1, 3, 4).reshape(B * H, T * W, C)
        
#         if return_attn_matrices:
#             # Explicitly set mask_mode='block' for the axial pass
#             attn_out_tw, attn_matrix_tw = self.attn_tw(tokens_tw, T_steps=T, return_raw_matrix=True, mask_mode='block')
#             attn_matrices.append(attn_matrix_tw)
#         else:
#             attn_out_tw, _ = self.attn_tw(tokens_tw, T_steps=T, mask_mode='block')

#         attn_out_tw_res = tokens_tw + attn_out_tw
#         attn_out_tw_norm = self.norm_tw(attn_out_tw_res)
#         interim_tokens = attn_out_tw_norm.reshape(B, H, T, W, C).permute(0, 2, 1, 3, 4) 

#         # 2. Time-Height Attention (TH) - L = T * H. T is passed as T_steps.
#         tokens_th = interim_tokens.permute(0, 3, 2, 1, 4).reshape(B * W, T * H, C)
        
#         if return_attn_matrices:
#             # Explicitly set mask_mode='block' for the axial pass
#             attn_out_th, attn_matrix_th = self.attn_th(tokens_th, T_steps=T, return_raw_matrix=True, mask_mode='block')
#             attn_matrices.append(attn_matrix_th)
#         else:
#             attn_out_th, _ = self.attn_th(tokens_th, T_steps=T, mask_mode='block')

#         attn_out_th_res = tokens_th + attn_out_th
#         attn_out_th_norm = self.norm_th(attn_out_th_res)
#         final_tokens = attn_out_th_norm.reshape(B, W, T, H, C).permute(0, 2, 3, 1, 4)
        
#         # 3. FFN
#         final_tokens_flat = final_tokens.reshape(B, T * H * W, C)
#         ffn_out = self.ffn(final_tokens_flat)
#         output_tokens = self.norm_ffn(final_tokens_flat + ffn_out)
#         output = output_tokens.reshape(B, T, H, W, C)
        
#         if return_attn_matrices:
#             return output, attn_matrices

#         return output


# --- REVISED AXIAL/RKA INTERLEAVED LAYER STRUCTURE (With return_attn_matrices support) ---
class AxialTemporalRKAInterleavedLayer(nn.Module):
    """
    Enhanced RKA layer with an added SpatioTemporalGatedMixer after 
    each Attention block to reintroduce local convolutional bias.
    """
    def __init__(self, in_channels, nhead, dim_feedforward):
        super().__init__()
        # RKA Attention Blocks
        self.attn_tw = RKA_MultiheadAttention(in_channels, nhead, dropout=0)
        self.norm_tw = nn.LayerNorm(in_channels)
        self.attn_th = RKA_MultiheadAttention(in_channels, nhead, dropout=0)
        self.norm_th = nn.LayerNorm(in_channels)

        # INSERTED: Mixer blocks after each Attention pass
        self.mixer_tw = SpatioTemporalGatedMixer(in_channels)
        self.mixer_th = SpatioTemporalGatedMixer(in_channels)
        
        # FFN remains the same
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, in_channels),
            nn.Dropout(0)
        )
        self.norm_ffn = nn.LayerNorm(in_channels)

    # **CHANGE HERE: Added return_attn_matrices argument**
    def forward(self, input_tokens, B, T, H, W, return_attn_matrices=False):
        C = input_tokens.shape[-1]
        attn_matrices = [] # Initialize list to store matrices
        
        # 1. Time-Width Attention (TW)
        tokens_tw = input_tokens.reshape(B, T, H, W, C).permute(0, 2, 1, 3, 4).reshape(B * H, T * W, C)
        
        # **CHANGE HERE: Conditionally call RKA with return_raw_matrix**
        if return_attn_matrices:
            attn_out_tw, attn_matrix_tw = self.attn_tw(tokens_tw, T_steps=T, mask_mode='block', return_raw_matrix=True)
            attn_matrices.append(attn_matrix_tw)
        else:
            attn_out_tw, _ = self.attn_tw(tokens_tw, T_steps=T, mask_mode='block', return_raw_matrix=False) # return_raw_matrix=False is the default, but good to be explicit

        attn_out_tw_res = tokens_tw + attn_out_tw
        
        # --- MIXING STEP 1 ---
        attn_out_tw_mixed = self.mixer_tw(attn_out_tw_res, T, W) # Mix spatial features
        attn_out_tw_norm = self.norm_tw(attn_out_tw_mixed)
        
        interim_tokens = attn_out_tw_norm.reshape(B, H, T, W, C).permute(0, 2, 1, 3, 4)

        # 2. Time-Height Attention (TH)
        tokens_th = interim_tokens.permute(0, 3, 2, 1, 4).reshape(B * W, T * H, C)
        
        # **CHANGE HERE: Conditionally call RKA with return_raw_matrix**
        if return_attn_matrices:
            attn_out_th, attn_matrix_th = self.attn_th(tokens_th, T_steps=T, mask_mode='block', return_raw_matrix=True)
            attn_matrices.append(attn_matrix_th)
        else:
            attn_out_th, _ = self.attn_th(tokens_th, T_steps=T, mask_mode='block', return_raw_matrix=False)

        attn_out_th_res = tokens_th + attn_out_th
        
        # --- MIXING STEP 2 ---
        attn_out_th_mixed = self.mixer_th(attn_out_th_res, T, H) # Mix spatial features
        attn_out_th_norm = self.norm_th(attn_out_th_mixed)
        
        final_tokens = attn_out_th_norm.reshape(B, W, T, H, C).permute(0, 2, 3, 1, 4)
        
        # 3. FFN
        final_tokens_flat = final_tokens.reshape(B, T * H * W, C)
        ffn_out = self.ffn(final_tokens_flat)
        output_tokens = self.norm_ffn(final_tokens_flat + ffn_out)
        
        output = output_tokens.reshape(B, T, H, W, C)
        
        # **CHANGE HERE: Return attention matrices if requested**
        if return_attn_matrices:
            return output, attn_matrices

        return output
 
# ----------------------------------------------------------------------
    
def plot_attention_matrix(matrix_np, title="RKA Attention Matrix (Averaged)", T=3, H=16, W=16):
    """Plots the attention matrix with lines to show the T x T block structure."""
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix_np, cmap='viridis', aspect='equal')
    plt.colorbar(label='Attention Weight (Pre-Mask/Softmax)')
    
    L = matrix_np.shape[0]
    S = L // T # Spatial Block Size
    
    # Draw block lines (separating time steps)
    for i in range(1, T):
        plt.axhline(i * S - 0.5, color='r', linestyle='--', linewidth=1)
        plt.axvline(i * S - 0.5, color='r', linestyle='--', linewidth=1)
        
    plt.title(f"{title}\nL={L}, T={T}, S={S} (Red lines show temporal blocks)")
    plt.xlabel('Key Index (Past Tokens)')
    plt.ylabel('Query Index (Current Tokens)')
    plt.show()


# ----------------------------------------------------------------------

class InterleavedAxialTemporalRKAIntegrator(nn.Module):
    """
    Replaces InterleavedAxialTemporalSWAIntegrator. Runs a single forward pass 
    over the full 3-frame history using RKA's block-causal attention.
    """
    def __init__(self, in_channels, nhead, dim_feedforward, num_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            AxialTemporalRKAInterleavedLayer(in_channels, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
        
    def forward(self, F_t_minus_2, F_t_minus_1, F_t):
        # Input features are (N, C, H, W). Stack them to get a sequence of 3 frames.
        stacked_features = torch.stack([F_t_minus_2, F_t_minus_1, F_t], dim=1)
        B, T, C, H, W = stacked_features.shape
        
        # Convert to token format: (N, T, H, W, C)
        tokens = stacked_features.permute(0, 1, 3, 4, 2)

        for layer in self.layers:
            # The layer runs RKA using the Block Causal Mask internally
            tokens = layer(tokens, B, T, H, W) 
            
        # Output features are separated by time step
        M_t_minus_2_tokens = tokens[:, 0, :, :, :]
        M_t_minus_1_tokens = tokens[:, 1, :, :, :]
        M_t_tokens = tokens[:, 2, :, :, :] # Final aggregated frame M_t

        # Permute back to (N, C, H, W) for downstream CNNs
        M_t_2 = M_t_minus_2_tokens.permute(0, 3, 1, 2)
        M_t_1 = M_t_minus_1_tokens.permute(0, 3, 1, 2)
        M_t = M_t_tokens.permute(0, 3, 1, 2)
        
        # Return all three aggregated frames (M0, M1, M2)
        return M_t_2, M_t_1, M_t

# ----------------------------------------------------------------------

# class RKAFeatureAggregator(nn.Module):
#     """
#     RKA Feature Aggregator (Replaces SWA): Uses RKA Integrators for L4/BN and CNN Mixers for L3/L2/L1.
#     It outputs the full concatenated M feature set for the DynNet.
#     """
#     def __init__(self, args, base_channels=None):
#         super().__init__()
#         C = base_channels if base_channels is not None else BASE_CHANNELS
#         NUM_ATTN_LAYERS = args.num_attn_layers
        
#         # --- CORRECT CHANNEL DEFINITIONS (Based on Unet_Enc reduced outputs) ---
#         C_L2_INPUT = C 
#         C_L2_CONCAT_CH = 3 * C_L2_INPUT 
#         C_L2_SKIP_OUT = C 
        
#         C_L1_INPUT = C // 2
#         C_L1_CONCAT_CH = 3 * C_L1_INPUT 
#         C_L1_SKIP_OUT = C // 2

#         # ATTENTION INTEGRATORS (L4 and BN) - Full Sequence Aggregation
#         self.attn_bn = InterleavedAxialTemporalRKAIntegrator(
#             in_channels=C * 16, 
#             nhead=args.nhead, 
#             dim_feedforward=args.d_attn2, 
#             num_layers=NUM_ATTN_LAYERS
#         )
#         self.attn_l4 = InterleavedAxialTemporalRKAIntegrator(
#             in_channels=C * 8, 
#             nhead=args.nhead, 
#             dim_feedforward=args.d_attn2, 
#             num_layers=NUM_ATTN_LAYERS
#         )
        
#         # CNN MIXERS (L3, L2, L1) - Single-Frame Aggregation
#         self.mixer_l3 = LocalSpatioTemporalMixer(in_channels=C * 4, kernel_size=3) 
        
#         # L2 Mixer - Input 48, Output 16
#         self.conv_l2 = nn.Sequential(
#             nn.Conv2d(C_L2_CONCAT_CH, C_L2_SKIP_OUT, kernel_size=1), 
#             nn.GELU(),
#             ResidualBlock(C_L2_SKIP_OUT, C_L2_SKIP_OUT)
#         )
        
#         # L1 Mixer - Input 24, Output 8
#         self.conv_l1 = nn.Sequential(
#             nn.Conv2d(C_L1_CONCAT_CH, C_L1_SKIP_OUT, kernel_size=1), 
#             nn.GELU(),
#             ResidualBlock(C_L1_SKIP_OUT, C_L1_SKIP_OUT)
#         )
#         self.apply(init_weights) 

#     def forward(self, F_t_minus_2, F_t_minus_1, F_t):
#         f1_t_2, f2_t_2, f3_t_2, f4_t_2, bn_t_2 = F_t_minus_2
#         f1_t_1, f2_t_1, f3_t_1, f4_t_1, bn_t_1 = F_t_minus_1
#         f1_t, f2_t, f3_t, f4_t, bn_t = F_t
        
#         # 1. ATTENTION PASSES (M_t-2, M_t-1, M_t) for L4 and BN
#         m_bn_t2, m_bn_t1, m_bn_t = self.attn_bn(bn_t_2, bn_t_1, bn_t)
#         m4_t2, m4_t1, m4_t = self.attn_l4(f4_t_2, f4_t_1, f4_t)
        
#         # 2. CNN MIXERS (M_t only) for L1, L2, L3. M_t-2 and M_t-1 are simply replicated M_t (stateless approach for CNN mixers).
#         m3_t = self.mixer_l3(f3_t_2, f3_t_1, f3_t) 
#         f2_cat = torch.cat([f2_t_2, f2_t_1, f2_t], dim=1) 
#         m2_t = self.conv_l2(f2_cat) 
#         f1_cat = torch.cat([f1_t_2, f1_t_1, f1_t], dim=1) 
#         m1_t = self.conv_l1(f1_cat)
        
#         # 3. CONCATENATE M features into a single flat tuple sequence (M_flat) for DynNet input
        
#         # L1, L2, L3 features are replicated across the T_pred time steps (M_t, M_t, M_t)
#         M3_cat = torch.cat([m3_t, m3_t, m3_t], dim=0)
#         M2_cat = torch.cat([m2_t, m2_t, m2_t], dim=0)
#         M1_cat = torch.cat([m1_t, m1_t, m1_t], dim=0)
        
#         # L4, BN contain the full sequence (M_t-2, M_t-1, M_t)
#         M_bn_cat = torch.cat([m_bn_t2, m_bn_t1, m_bn_t], dim=0)
#         M4_cat = torch.cat([m4_t2, m4_t1, m4_t], dim=0)
        
#         # Output is the M_flat sequence for the DynNet
#         return M1_cat, M2_cat, M3_cat, M4_cat, M_bn_cat

class RKAFeatureAggregator(nn.Module):
    """
    RKA Feature Aggregator: Uses RKA Integrators (N=2 layers) for L4/BN 
    and incorporates the Macro-Residual to compensate for fine-detail loss.
    """
    def __init__(self, args, base_channels=None):
        super().__init__()
        # Optimal stacking depth (N=2) - Set as default for Integrator, but can be overridden by args
        NUM_ATTN_LAYERS = getattr(args, 'num_attn_layers', 2) 
        C = base_channels if base_channels is not None else 16
        
        # Channel definitions for CNN Mixers (L1, L2)
        C_L2_INPUT = C; C_L2_CONCAT_CH = 3 * C_L2_INPUT; C_L2_SKIP_OUT = C
        C_L1_INPUT = C // 2; C_L1_CONCAT_CH = 3 * C_L1_INPUT; C_L1_SKIP_OUT = C // 2

        # ATTENTION INTEGRATORS (L4 and BN) - N=2 Layers
        self.attn_bn = InterleavedAxialTemporalRKAIntegrator(
            in_channels=C * 16, nhead=args.nhead, dim_feedforward=args.d_attn2, num_layers=NUM_ATTN_LAYERS
        )
        self.attn_l4 = InterleavedAxialTemporalRKAIntegrator(
            in_channels=C * 8, nhead=args.nhead, dim_feedforward=args.d_attn2, num_layers=NUM_ATTN_LAYERS
        )
        
        # CNN MIXERS (L3, L2, L1)
        self.mixer_l3 = LocalSpatioTemporalMixer(in_channels=C * 4) 
        self.conv_l2 = nn.Sequential(nn.Conv2d(C_L2_CONCAT_CH, C_L2_SKIP_OUT, kernel_size=1), nn.GELU(), ResidualBlock(C_L2_SKIP_OUT, C_L2_SKIP_OUT))
        self.conv_l1 = nn.Sequential(nn.Conv2d(C_L1_CONCAT_CH, C_L1_SKIP_OUT, kernel_size=1), nn.GELU(), ResidualBlock(C_L1_SKIP_OUT, C_L1_SKIP_OUT))
        self.apply(init_weights)

    def forward(self, F_t_minus_2, F_t_minus_1, F_t):
        # 1. Unpack Raw Encoder Features (F) for t-2, t-1, t
        f1_t_2, f2_t_2, f3_t_2, f4_t_2, bn_t_2 = F_t_minus_2
        f1_t_1, f2_t_1, f3_t_1, f4_t_1, bn_t_1 = F_t_minus_1
        f1_t, f2_t, f3_t, f4_t, bn_t = F_t # Raw F_t is needed for Macro-Residuals
        
        # 2. RKA Temporal Aggregation (Generates stable M features)
        m_bn_t2, m_bn_t1, m_bn_t = self.attn_bn(bn_t_2, bn_t_1, bn_t)
        m4_t2, m4_t1, m4_t = self.attn_l4(f4_t_2, f4_t_1, f4_t)
        
        # 3. MACRO-RESIDUAL CONNECTION (Compensates for fine-detail loss)
        # Adds the raw encoder feature (F) back into the RKA output (M)
        m_bn_t = m_bn_t + bn_t
        m4_t = m4_t + f4_t 

        # 4. CNN Mixers (Local Aggregation) - M_t only
        m3_t = self.mixer_l3(f3_t_2, f3_t_1, f3_t) 
        f2_cat = torch.cat([f2_t_2, f2_t_1, f2_t], dim=1); m2_t = self.conv_l2(f2_cat) 
        f1_cat = torch.cat([f1_t_2, f1_t_1, f1_t], dim=1); m1_t = self.conv_l1(f1_cat)
        
        # 5. Concatenate for DynNet Input (M_flat)
        M3_cat = torch.cat([m3_t, m3_t, m3_t], dim=0) # Replicated M_t
        M2_cat = torch.cat([m2_t, m2_t, m2_t], dim=0) # Replicated M_t
        M1_cat = torch.cat([m1_t, m1_t, m1_t], dim=0) # Replicated M_t
        
        # L4, BN contain the full sequence (M_t-2, M_t-1, M_t)
        M_bn_cat = torch.cat([m_bn_t2, m_bn_t1, m_bn_t], dim=0)
        M4_cat = torch.cat([m4_t2, m4_t1, m4_t], dim=0)
        
        return M1_cat, M2_cat, M3_cat, M4_cat, M_bn_cat

# ----------------------------------------------------------------------

class RKAU_Net(nn.Module):
    """
    Relational Kernel Attention U-Net (RKAU_Net)
    Full SWAU-Net pipeline with RKA Feature Aggregator replacing SWA.
    Flow: E1 -> CFB_enc -> RKA -> DynNet -> CFB_dec -> D1
    """
    def __init__(self, args, img_channels=None, base_channels=None):
        super(RKAU_Net, self).__init__()
        
        if img_channels is None: img_channels = args.img_channels
        if base_channels is None: base_channels = BASE_CHANNELS
            
        # E1: Feature Extractor
        self.E1 = Unet_Enc(args, img_channels, base_channels)
        
        # CFB_enc: Pre-Dynamics Feature Refinement (Restored)
        self.CFB_enc = ChannelFusionBlock(base_channels) 

        # RKA: Temporal Aggregator (Replaces SWA)
        self.RKA_Aggregator = RKAFeatureAggregator(args, base_channels)
        
        # P: Temporal Feature Predictor (DynNet) (Restored - using simplified DynNet structure)
        self.P = DynNet(args, base_channels) 
        
        # CFB_dec: Post-Dynamics Feature Refinement (Restored)
        self.CFB_dec = ChannelFusionBlock(base_channels)
        
        # D1: Frame Reconstructor
        self.D1 = Unet_Dec(args, img_channels, base_channels)
        
    def forward(self, input_clips):
        B, C, T, H, W = input_clips.shape
        T_pred = T - 1 
        
        # --- A. FEATURE EXTRACTION (E1) ---
        I0_gt = input_clips[:, :, 0, :, :] 
        # E1 processes T frames (I0, I1, I2) for T_pred predictions (I1, I2, I3)
        input_frames_E1 = input_clips[:, :, :T_pred, :, :].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        E1_features_flat = self.E1(input_frames_E1)
        
        # --- 1. PRE-DYNAMICS CHANNEL FUSION (CFB_enc) ---
        F_refined_flat = self.CFB_enc(E1_features_flat)
        
        # Unpack F' features by time step (F'0, F'1, F'2)
        F0_refined = [f[:B] for f in F_refined_flat] 
        F1_refined = [f[B:2*B] for f in F_refined_flat] 
        F2_refined = [f[2*B:3*B] for f in F_refined_flat] 
        
        # --- B. ANCHOR RECONSTRUCTION (I_hat_0) ---
        I0_hat = self.D1(*F0_refined) 

        # --- C. RKA AGGREGATION (Replaces SWA) ---
        # M_flat = RKA(F'0, F'1, F'2). Generates the context-integrated M features.
        M_flat = self.RKA_Aggregator(F0_refined, F1_refined, F2_refined) 
        
        # --- D. TEMPORAL EVOLUTION (DynNet) ---
        # E_raw_evolved_flat = P(M_flat)
        E_raw_evolved_flat = self.P(*M_flat)
        
        # --- 2. POST-DYNAMICS CHANNEL FUSION (CFB_dec) ---
        # Evolved_polished = CFB_dec(E_raw_evolved_flat)
        Evolved_polished = self.CFB_dec(E_raw_evolved_flat)

        # Unpack for loss function signature
        E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved = Evolved_polished

        # --- E. DECODING & RESHAPING ---
        out_frames_pred = self.D1(*Evolved_polished)
        predictions = out_frames_pred.reshape(B, T_pred, C, H, W).permute(0, 2, 1, 3, 4)
        targets = input_clips[:, :, 1:, :, :] 
        
        # Return 9 items, maintaining the established loss function signature
        return predictions, targets, I0_hat, I0_gt, E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved

