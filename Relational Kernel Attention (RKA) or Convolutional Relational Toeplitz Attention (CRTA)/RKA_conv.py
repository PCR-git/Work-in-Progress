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
    
# -----------------------------------------------------------------------------------------
       
class RKA_MultiheadAttention_Fast(nn.Module):
    """
    Relational Kernel Attention (RKA) Head - Fast Convolutional Implementation.
    
    The Q_rel generation now uses a strictly CAUSAL convolution to prevent
    future information leakage into the global stationary kernel.
    """
    def __init__(self, embed_dim, num_heads, dropout=0., temporal_kernel_size=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.temporal_kernel_size = temporal_kernel_size
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        
        # Linear projection for Q, K, V
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)  
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Layers for Learned fusion - MODIFIED
        # Causal convolution requires manual padding or using padding='causal'.
        # We use padding=0 here and handle the padding manually in forward for strict control.
        self.q_rel_conv1d = nn.Conv1d(in_channels=self.head_dim,
                                      out_channels=self.head_dim,
                                      kernel_size=temporal_kernel_size,
                                      padding=0, # No automatic padding
                                      groups=self.head_dim)
        self.q_rel_gelu = nn.GELU()
        self.q_rel_norm = nn.LayerNorm(self.head_dim)
        
        self.apply(init_weights)

    def _reshape_heads(self, t):
        N, L, D = t.shape
        # Output shape for RKA operation: [N * H, L, D_H]
        return t.reshape(N, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(N * self.num_heads, L, self.head_dim)

    def forward(self, x, T_steps=None, return_raw_matrix=False, mask_mode='standard'):
        
        N, L, D = x.shape
        qkv = self.in_proj(x); q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape Q, K, V for Head calculation: [N*H, L, D_H]
        q_head = self._reshape_heads(q); k_head = self._reshape_heads(k); v_head = self._reshape_heads(v)
        N_H = q_head.shape[0] 
        D_H = self.head_dim      

        # --- O(LD) KERNEL COMPUTATION (with CAUSAL Temporal Mixing in Q_rel) ---
        # q_temp: [N*H, L, D_H] -> [N*H, D_H, L] (ready for Conv1D over time L)
        q_temp = q_head.transpose(1, 2)  
        
        # 1. Enforce Causality for Q_rel (Input Padding)
        k_t = self.temporal_kernel_size
        q_padding_needed = k_t - 1
        
        # Pad on the left
        q_padded = F.pad(q_temp, (q_padding_needed, 0))

        # 2. Apply temporal Conv1D (k_t > 1) - Output is length L
        q_temp = self.q_rel_conv1d(q_padded) 
        q_temp = self.q_rel_gelu(q_temp)
        
        # 3. LayerNorm and Pooling
        q_temp = q_temp.transpose(1, 2) # [N*H, L, D_H]
        q_temp = self.q_rel_norm(q_temp)

        # Average over all L tokens to get the single global relational query [N*H, 1, D_H]
        q_rel_single = q_temp.mean(dim=1, keepdim=True) 
        
        # 4. Compute the Relational Kernel vector: W_kernel [N*H, 1, L]
        W_kernel = torch.matmul(q_rel_single * self.scaling, k_head.transpose(-2, -1))
        W_kernel_row = W_kernel.squeeze(1) # [N*H, L]
        
        # 5. Kernel Normalization (L1 norm)
        W_kernel_normalized = W_kernel_row / (W_kernel_row.abs().sum(dim=1, keepdim=True) + 1e-6)
        
        # --- O(L log L) / O(LK) CAUSAL CONVOLUTION ---
        K = L # Kernel size is the full sequence length L
        
        # --- Prepare Tensors for Grouped Conv1D ---
        V_input = v_head.transpose(1, 2) # [N*H, D_H, L]

        # 1. FLIP THE KERNEL (Essential for F.conv1d causal alignment)
        W_kernel_flipped = torch.flip(W_kernel_normalized, dims=[-1])
        
        # 2. Tile and Reshape Weight W: [N*H*D_H, 1, L]
        W_kernel_tiled = W_kernel_flipped.unsqueeze(1).repeat(1, D_H, 1).reshape(N_H * D_H, 1, L) 
        
        # 3. Reshape Input V: [1, N*H*D_H, L]
        V_signal_final = V_input.reshape(1, N_H * D_H, L)

        # 4. Apply Causal Convolution
        padding_needed = K - 1 # Standard causal padding
        groups_final = N_H * D_H
        
        # Pad on the left
        padded_input = F.pad(V_signal_final, (padding_needed, 0))
        
        # Convolution. The output size is exactly L.
        attn_output_flat = F.conv1d(
            input=padded_input,
            weight=W_kernel_tiled,
            groups=groups_final
        )
        
        # 5. Reshape and Final Projection
        attn_output = attn_output_flat.reshape(N_H, D_H, L).transpose(1, 2)
        
        attn_output_reshaped = attn_output.reshape(N, self.num_heads, L, self.head_dim).permute(0, 2, 1, 3).flatten(start_dim=-2)
        output = self.out_proj(attn_output_reshaped)
        
        # Return mock weights for API consistency
        mock_attn_weights = torch.ones(N, L, device=x.device) / L 
        
        return output, mock_attn_weights

# ------------------------------------------------------------------------
 

class AxialTemporalRKAInterleavedLayer_Fast(nn.Module):
    """
    Enhanced RKA layer using the FAST Convolutional RKA Head, adapted for Block Causality.
    
    The input is now reshaped such that RKA_MultiheadAttention_Fast only sees 
    the temporal dimension T as its sequence length L, preserving spatial mixing.
    """
    def __init__(self, in_channels, nhead, dim_feedforward):
        super().__init__()
        # RKA Attention Blocks: Replaced RKA_MultiheadAttention with RKA_MultiheadAttention_Fast
        self.attn_tw = RKA_MultiheadAttention_Fast(in_channels, nhead, dropout=0)
        self.norm_tw = nn.LayerNorm(in_channels)
        self.attn_th = RKA_MultiheadAttention_Fast(in_channels, nhead, dropout=0)
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

    def forward(self, input_tokens, B, T, H, W):
        C = input_tokens.shape[-1]
        
        # 1. Time-Attention (TW) - Sequence length L = T
        # Reshape: (B, T, H, W, C) -> (B * H * W, T, C)
        # This treats every spatial token (H*W) as an independent sequence of length T.
        tokens_tw_in = input_tokens.reshape(B * H * W, T, C)
        
        # RKA runs on L=T, enforcing temporal causality
        attn_out_tw, _ = self.attn_tw(tokens_tw_in)
        attn_out_tw_res = tokens_tw_in + attn_out_tw
        
        # Reshape back to (B, H, W, T, C), then permute to (B, T, H, W, C) for Mixer
        attn_out_tw_res_unflat = attn_out_tw_res.reshape(B, H, W, T, C).permute(0, 3, 1, 2, 4)
        
        # Re-flatten across W for the mixer: (B * T * H, W, C) - Sequence length is W
        attn_out_tw_mixed_flat = attn_out_tw_res_unflat.reshape(B * T * H, W, C)

        # --- MIXING STEP 1 (Spatial Mixing along W) ---
        # FIX: The sequence length is W. It should be factored as H_out=1, W_out=W for 1D mixing.
        attn_out_tw_mixed = self.mixer_tw(attn_out_tw_mixed_flat, 1, W) 
        
        attn_out_tw_norm = self.norm_tw(attn_out_tw_mixed)
        
        # Reshape back to (B, T, H, W, C) structure
        interim_tokens = attn_out_tw_norm.reshape(B, T, H, W, C)

        # 2. Time-Attention (TH) - Sequence length L = T
        # Reshape: (B, T, H, W, C) -> Permute (B, W, H, T, C) -> Reshape (B * W * H, T, C)
        tokens_th_in = interim_tokens.permute(0, 3, 2, 1, 4).reshape(B * W * H, T, C) 

        # RKA runs on L=T, enforcing temporal causality
        attn_out_th, _ = self.attn_th(tokens_th_in)
        attn_out_th_res = tokens_th_in + attn_out_th
        
        # Reshape back to (B, W, H, T, C), then permute to (B, T, H, W, C) structure
        attn_out_th_res_unflat = attn_out_th_res.reshape(B, W, H, T, C).permute(0, 3, 2, 1, 4)

        # Re-flatten across H for the mixer: (B * T * W, H, C) - Sequence length is H
        attn_out_th_mixed_flat = attn_out_th_res_unflat.permute(0, 3, 1, 2, 4).reshape(B * T * W, H, C)
        
        # --- MIXING STEP 2 (Spatial Mixing along H) ---
        # FIX: The sequence length is H. It should be factored as H_out=H, W_out=1 for 1D mixing.
        attn_out_th_mixed = self.mixer_th(attn_out_th_mixed_flat, H, 1) 
        
        attn_out_th_norm = self.norm_th(attn_out_th_mixed)
        
        final_tokens_unflat = attn_out_th_norm.reshape(B, T, H, W, C)
        
        # 3. FFN
        final_tokens_flat = final_tokens_unflat.reshape(B, T * H * W, C)
        ffn_out = self.ffn(final_tokens_flat)
        output_tokens = self.norm_ffn(final_tokens_flat + ffn_out)
        
        return output_tokens.reshape(B, T, H, W, C)
    
# -----------------------------------------------------------------------------------------
      
    
# --- 1. Fast RKA Integrator (Wrapper for the Fast Axial Layer) ---

class InterleavedAxialTemporalRKAIntegrator_Fast(nn.Module):
    """
    Fast version of the RKA Integrator, using the O(L log L) Fast Axial Layer.

    This runs a single forward pass over the input clip (F_t-2, F_t-1, F_t)
    using the Block Causal approach factored into the Fast Axial Layer.
    """
    def __init__(self, in_channels, nhead, dim_feedforward, num_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        
        # KEY CHANGE: Use the Fast Axial Layer for block causality
        self.layers = nn.ModuleList([
            AxialTemporalRKAInterleavedLayer_Fast(in_channels, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
        
    def forward(self, F_t_minus_2, F_t_minus_1, F_t):
        # Input features are (N, C, H, W). Stack them to get a sequence of 3 frames.
        stacked_features = torch.stack([F_t_minus_2, F_t_minus_1, F_t], dim=1)
        B, T, C, H, W = stacked_features.shape
        
        # Convert to token format: (N, T, H, W, C)
        tokens = stacked_features.permute(0, 1, 3, 4, 2)

        for layer in self.layers:
            # The layer runs RKA using the Block Causal logic internally
            tokens = layer(tokens, B, T, H, W) 
            
        # Output features are separated by time step
        M_t_minus_2_tokens = tokens[:, 0, :, :, :]
        M_t_minus_1_tokens = tokens[:, 1, :, :, :]
        M_t_tokens = tokens[:, 2, :, :, :] 

        # Permute back to (N, C, H, W) for downstream CNNs
        M_t_2 = M_t_minus_2_tokens.permute(0, 3, 1, 2)
        M_t_1 = M_t_minus_1_tokens.permute(0, 3, 1, 2)
        M_t = M_t_tokens.permute(0, 3, 1, 2)
        
        # Return all three aggregated frames (M0, M1, M2)
        return M_t_2, M_t_1, M_t

# --- 2. Fast RKA Feature Aggregator ---

class RKAFeatureAggregator_Fast(nn.Module):
    """
    Fast RKA Feature Aggregator, incorporating Macro-Residuals and the new Fast Integrator.
    
    This class orchestrates feature mixing, using fast integrators for global context 
    (L4, BN) and CNN mixers for local context (L3, L2, L1).
    """
    def __init__(self, args, base_channels=None):
        super().__init__()
        NUM_ATTN_LAYERS = getattr(args, 'num_attn_layers', 2) 
        C = base_channels if base_channels is not None else 16
        
        # Channel definitions (Assuming necessary helpers like BASE_CHANNELS are defined)
        C_L2_INPUT = C; C_L2_CONCAT_CH = 3 * C_L2_INPUT; C_L2_SKIP_OUT = C
        C_L1_INPUT = C // 2; C_L1_CONCAT_CH = 3 * C_L1_INPUT; C_L1_SKIP_OUT = C // 2

        # KEY CHANGE: Use the Fast Integrator
        self.attn_bn = InterleavedAxialTemporalRKAIntegrator_Fast(
            in_channels=C * 16, nhead=args.nhead, dim_feedforward=args.d_attn2, num_layers=NUM_ATTN_LAYERS
        )
        self.attn_l4 = InterleavedAxialTemporalRKAIntegrator_Fast(
            in_channels=C * 8, nhead=args.nhead, dim_feedforward=args.d_attn2, num_layers=NUM_ATTN_LAYERS
        )
        
        # CNN MIXERS (L3, L2, L1) - Local Aggregation (Assumes LocalSpatioTemporalMixer, ResidualBlock are available)
        self.mixer_l3 = LocalSpatioTemporalMixer(in_channels=C * 4) 
        self.conv_l2 = nn.Sequential(nn.Conv2d(C_L2_CONCAT_CH, C_L2_SKIP_OUT, kernel_size=1), nn.GELU(), ResidualBlock(C_L2_SKIP_OUT, C_L2_SKIP_OUT))
        self.conv_l1 = nn.Sequential(nn.Conv2d(C_L1_CONCAT_CH, C_L1_SKIP_OUT, kernel_size=1), nn.GELU(), ResidualBlock(C_L1_SKIP_OUT, C_L1_SKIP_OUT))
        
        # Assumes init_weights is available
        # self.apply(init_weights) 

    def forward(self, F_t_minus_2, F_t_minus_1, F_t):
        # 1. Unpack Raw Encoder Features (F) for t-2, t-1, t
        f1_t_2, f2_t_2, f3_t_2, f4_t_2, bn_t_2 = F_t_minus_2
        f1_t_1, f2_t_1, f3_t_1, f4_t_1, bn_t_1 = F_t_minus_1
        f1_t, f2_t, f3_t, f4_t, bn_t = F_t 
        
        # 2. RKA Temporal Aggregation (Generates stable M features using Fast Integrator)
        m_bn_t2, m_bn_t1, m_bn_t = self.attn_bn(bn_t_2, bn_t_1, bn_t)
        m4_t2, m4_t1, m4_t = self.attn_l4(f4_t_2, f4_t_1, f4_t)
        
        # 3. MACRO-RESIDUAL CONNECTION
        m_bn_t = m_bn_t + bn_t
        m4_t = m4_t + f4_t 

        # 4. CNN Mixers (Local Aggregation) - M_t only
        m3_t = self.mixer_l3(f3_t_2, f3_t_1, f3_t) 
        f2_cat = torch.cat([f2_t_2, f2_t_1, f2_t], dim=1); m2_t = self.conv_l2(f2_cat) 
        f1_cat = torch.cat([f1_t_2, f1_t_1, f1_t], dim=1); m1_t = self.conv_l1(f1_cat)
        
        # 5. Concatenate for DynNet Input (M_flat)
        # L1, L2, L3 features are replicated across the T_pred time steps (M_t, M_t, M_t)
        M3_cat = torch.cat([m3_t, m3_t, m3_t], dim=0) 
        M2_cat = torch.cat([m2_t, m2_t, m2_t], dim=0) 
        M1_cat = torch.cat([m1_t, m1_t, m1_t], dim=0) 
        
        # L4, BN contain the full sequence (M_t-2, M_t-1, M_t)
        M_bn_cat = torch.cat([m_bn_t2, m_bn_t1, m_bn_t], dim=0)
        M4_cat = torch.cat([m4_t2, m4_t1, m4_t], dim=0)
        
        return M1_cat, M2_cat, M3_cat, M4_cat, M_bn_cat
    
    
    
class RKAU_Net_Fast(nn.Module):
    """
    Relational Kernel Attention U-Net (RKAU_Net) - Fast Version.
    Uses the O(L log L) convolutional RKA feature integration pipeline.
    """
    def __init__(self, args, img_channels=None, base_channels=None):
        super(RKAU_Net_Fast, self).__init__()
        
        if img_channels is None: img_channels = args.img_channels
        if base_channels is None: base_channels = BASE_CHANNELS
        
        # E1: Feature Extractor
        self.E1 = Unet_Enc(args, img_channels, base_channels)
        
        # CFB_enc: Pre-Dynamics Feature Refinement 
        self.CFB_enc = ChannelFusionBlock(base_channels) 

        # KEY CHANGE: RKA: Temporal Aggregator (Now using the Fast Aggregator)
        self.RKA_Aggregator = RKAFeatureAggregator_Fast(args, base_channels)
        
        # P: Temporal Feature Predictor (DynNet)
        self.P = DynNet(args, base_channels) 
        
        # CFB_dec: Post-Dynamics Feature Refinement 
        self.CFB_dec = ChannelFusionBlock(base_channels)
        
        # D1: Frame Reconstructor
        self.D1 = Unet_Dec(args, img_channels, base_channels)
        
    def forward(self, input_clips):
        B, C, T, H, W = input_clips.shape
        T_pred = T - 1 
        
        # --- A. FEATURE EXTRACTION (E1) ---
        I0_gt = input_clips[:, :, 0, :, :] 
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

        # --- C. RKA AGGREGATION (Fast Version) ---
        M_flat = self.RKA_Aggregator(F0_refined, F1_refined, F2_refined) 
        
        # --- D. TEMPORAL EVOLUTION (DynNet) ---
        E_raw_evolved_flat = self.P(*M_flat)
        
        # --- 2. POST-DYNAMICS CHANNEL FUSION (CFB_dec) ---
        Evolved_polished = self.CFB_dec(E_raw_evolved_flat)

        # Unpack for loss function signature
        E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved = Evolved_polished

        # --- E. DECODING & RESHAPING ---
        out_frames_pred = self.D1(*Evolved_polished)
        predictions = out_frames_pred.reshape(B, T_pred, C, H, W).permute(0, 2, 1, 3, 4)
        targets = input_clips[:, :, 1:, :, :] 
        
        # Return 9 items, maintaining the established loss function signature
        return predictions, targets, I0_hat, I0_gt, E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved
    
    