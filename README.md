# Giải thích chi tiết toàn bộ LoopSplat

**Paper:** LoopSplat: Loop Closure by Registering 3D Gaussian Splats  
**Tác giả:** Liyuan Zhu, Yue Li, Erik Sandström, Shengyu Huang, Konrad Schindler, Iro Armeni  
**Hội nghị:** 3DV 2025 (Oral)  
**arXiv:** 2408.10154

---

# PHẦN 1: 3D GAUSSIAN SPLATTING (NỀN TẢNG)

## 1.1 Tại sao chọn 3DGS thay vì NeRF?

**NeRF (Neural Radiance Fields):**

```
Input: 3D point + view direction
       |
       v
   MLP Network (nhiều layers)
       |
       v
Output: color + density
```

**Vấn đề của NeRF:**
- Rendering chậm (cần query MLP hàng triệu lần per image)
- Training chậm (hours)
- Implicit representation → khó edit

**3DGS:**

```
Input: Camera pose
       |
       v
   Rasterization (GPU-parallel)
       |
       v
Output: Image (real-time)
```

**Ưu điểm của 3DGS:**
- Rendering real-time (100+ FPS)
- Training nhanh (minutes)
- Explicit representation → dễ manipulate

## 1.2 Gaussian Primitive chi tiết

Mỗi Gaussian là một "blob" 3D với các thuộc tính:

```
Gaussian G = {
    μ ∈ ℝ³,           // Position (x, y, z)
    Σ ∈ ℝ³ˣ³,         // Covariance (shape)
    α ∈ [0, 1],       // Opacity
    c = {c_l^m}       // Spherical Harmonics coefficients
}
```

**Visualization:**

```
         z
         |
         |    ╭───╮
         |   /     \    ← Gaussian "blob"
         |  |   ●   |   ← μ (center)
         |   \     /
         |    ╰───╯
         |
         └────────────── x
        /
       y
```

## 1.3 Covariance Matrix Σ

**Ý nghĩa hình học:**
- Σ định nghĩa hình dạng ellipsoid của Gaussian
- Eigenvectors → hướng của các trục
- Eigenvalues → độ dài các trục

**Parameterization để đảm bảo positive semi-definite:**

```
Σ = R · S · Sᵀ · Rᵀ
```

Trong đó:
- **R**: Rotation matrix (3x3), parameterized bằng quaternion q = (w, x, y, z)
- **S**: Scale matrix = diag(s_x, s_y, s_z)

**Quaternion to Rotation:**

```
        ┌ 1-2(y²+z²)    2(xy-wz)      2(xz+wy)   ┐
R(q) =  | 2(xy+wz)      1-2(x²+z²)    2(yz-wx)   |
        └ 2(xz-wy)      2(yz+wx)      1-2(x²+y²) ┘
```

**Ví dụ cụ thể:**

```
# Gaussian hình cầu (isotropic)
s_x = s_y = s_z = 0.1
R = I (identity)
→ Σ = 0.01 · I

# Gaussian hình ellipsoid dẹt
s_x = 0.2, s_y = 0.2, s_z = 0.05
R = rotation 45° around z-axis
→ Ellipsoid nằm nghiêng, dẹt theo z
```

## 1.4 Spherical Harmonics cho Color

**Vấn đề:** Màu sắc thay đổi theo góc nhìn (view-dependent effects như specular reflection)

**Giải pháp:** Dùng Spherical Harmonics (SH) để model view-dependent color

```
c(d) = Σ_{l=0}^{L} Σ_{m=-l}^{l} c_l^m · Y_l^m(d)
```

Trong đó:
- **d**: View direction (unit vector)
- **Y_l^m**: Spherical Harmonic basis functions
- **c_l^m**: Learned coefficients
- **L**: Maximum degree (thường L=3 → 16 coefficients per channel)

**SH Basis Functions (degree 0, 1, 2):**

```
Y_0^0 = 0.282095                           // Constant (ambient)

Y_1^{-1} = 0.488603 · y                    // Linear terms
Y_1^0 = 0.488603 · z
Y_1^1 = 0.488603 · x

Y_2^{-2} = 1.092548 · xy                   // Quadratic terms
Y_2^{-1} = 1.092548 · yz
Y_2^0 = 0.315392 · (3z² - 1)
Y_2^1 = 1.092548 · xz
Y_2^2 = 0.546274 · (x² - y²)
```

**Visualization:**

```
Degree 0 (constant):     Degree 1 (linear):      Degree 2 (quadratic):
      ●                    ◐ ◑ ◒                   More complex lobes
   (uniform)            (directional)
```

## 1.5 Splatting Process (3D → 2D)

**Step 1: Transform to Camera Space**

```
μ_cam = T_cw · μ_world

      ┌ R | t ┐   ┌ μ_x ┐     ┌ μ_cam_x ┐
      |       | · | μ_y |  =  | μ_cam_y |
      └ 0 | 1 ┘   | μ_z |     | μ_cam_z |
                  └  1  ┘     └    1    ┘
```

**Step 2: Project to 2D**

```
μ' = π(μ_cam)

     ┌ f_x · μ_cam_x / μ_cam_z + c_x ┐
μ' = |                                |
     └ f_y · μ_cam_y / μ_cam_z + c_y ┘
```

**Step 3: Project Covariance (EWA Splatting)**

3D covariance → 2D covariance qua Jacobian của projection:

```
Σ' = J · W · Σ · Wᵀ · Jᵀ
```

Trong đó:
- W = R_cw (camera rotation)
- J = Jacobian của perspective projection

```
      ∂π     ┌ f_x/z    0      -f_x·x/z² ┐
J = ───── = |                            |
     ∂p_cam └   0    f_y/z    -f_y·y/z² ┘
```

**Ví dụ số:**

```
Camera: f_x = f_y = 500, c_x = 320, c_y = 240

Gaussian at μ_cam = (1, 0.5, 5):
  μ'_x = 500 · 1/5 + 320 = 420
  μ'_y = 500 · 0.5/5 + 240 = 290

Jacobian at this point:
      ┌ 100    0    -20 ┐
  J = |                 |
      └  0    100   -10 ┘
```

## 1.6 Alpha Blending (Rendering)

**Sorting:** Gaussians được sort theo depth (front-to-back)

**Blending Formula:**

```
C(u) = Σ_{i=1}^{N} c_i · α_i · G_i(u) · T_i

where T_i = ∏_{j=1}^{i-1} (1 - α_j · G_j(u))
```

**Giải thích:**
- **c_i**: Color của Gaussian i
- **α_i**: Opacity của Gaussian i
- **G_i(u)**: Gaussian weight tại pixel u
- **T_i**: Transmittance (bao nhiêu light còn lại sau các Gaussians trước)

**Gaussian Weight:**

```
G_i(u) = exp(-½ · (u - μ'_i)ᵀ · Σ'_i⁻¹ · (u - μ'_i))
```

**Visualization của blending:**

```
Camera ──────────────────────────────────────▶ Scene

        G_1        G_2        G_3
         |          |          |
         v          v          v
       α_1=0.8    α_2=0.5    α_3=0.9
         |          |          |
         v          v          v
       
Pixel color = c_1·0.8 + c_2·0.5·(1-0.8) + c_3·0.9·(1-0.8)·(1-0.5) + ...
            = c_1·0.8 + c_2·0.1 + c_3·0.09 + ...
```

## 1.7 Depth Rendering

Tương tự color nhưng dùng depth thay vì color:

```
D(u) = Σ_{i=1}^{N} d_i · α_i · G_i(u) · T_i
```

Trong đó **d_i = μ_cam_z** là depth của Gaussian i.

---

# PHẦN 2: FRAME-TO-MODEL TRACKING

## 2.1 Mục tiêu

Cho:
- Current Gaussian map G
- New RGB-D frame (I, D)
- Initial pose guess T_init

Tìm: Camera pose T* tối ưu

## 2.2 Loss Function

```
L_track(T) = L_photo(T) + λ_geo · L_geo(T)
```

**Photometric Loss:**

```
L_photo = (1/|P|) · Σ_{u∈P} |I(u) - Î(u; T)|
```

Hoặc với SSIM:

```
L_photo = (1-λ) · L1 + λ · (1 - SSIM(I, Î))
```

**SSIM (Structural Similarity):**

```
SSIM(x, y) = (2μ_x·μ_y + C1)(2σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))
```

**Geometric Loss:**

```
L_geo = (1/|P|) · Σ_{u∈P} |D(u) - D̂(u; T)|
```

## 2.3 Optimization trên SE(3)

**SE(3) Manifold:**
- Không phải vector space → không thể dùng simple gradient descent
- Dùng Lie theory để parameterize

**Lie Algebra se(3):**

```
ξ = [ρ; φ] ∈ ℝ⁶

ρ = (ρ_1, ρ_2, ρ_3) ∈ ℝ³  // translation
φ = (φ_1, φ_2, φ_3) ∈ ℝ³  // rotation (axis-angle)
```

**Exponential Map (se(3) → SE(3)):**

```
T = exp(ξ^) = exp([φ^  ρ])
                 [0   0]
```

Trong đó φ^ là skew-symmetric matrix:

```
      ┌  0   -φ_3   φ_2 ┐
φ^ =  |  φ_3   0   -φ_1 |
      └ -φ_2  φ_1    0  ┘
```

**Closed-form exponential:**

```
exp(φ^) = I + (sin θ / θ) · φ^ + ((1 - cos θ) / θ²) · φ^²

where θ = ‖φ‖
```

**Full SE(3) exponential:**

```
          ┌ exp(φ^)  V·ρ ┐
exp(ξ^) = |              |
          └   0ᵀ      1  ┘

where V = I + ((1-cos θ)/θ²)·φ^ + ((θ-sin θ)/θ³)·φ^²
```

## 2.4 Optimization Algorithm

```python
def track_frame(G, I, D, T_init, max_iters=100, lr=0.01):
    """
    Track camera pose given new frame.
    """
    T = T_init
    
    for iter in range(max_iters):
        # Render from current pose
        I_hat, D_hat = render(G, T)
        
        # Compute loss
        L_photo = mean(abs(I - I_hat))
        L_geo = mean(abs(D - D_hat))
        L = L_photo + lambda_geo * L_geo
        
        # Compute gradient w.r.t. ξ (Lie algebra)
        grad_xi = compute_gradient(L, T)  # ∈ ℝ⁶
        
        # Update pose
        delta_xi = -lr * grad_xi
        delta_T = exp(delta_xi)  # se(3) → SE(3)
        T = delta_T @ T          # Left multiplication
        
        if converged(L):
            break
    
    return T
```

## 2.5 Jacobian Computation

**Gradient của loss w.r.t. ξ:**

```
∂L        ∂L     ∂Î(u)   ∂μ'    ∂(T·μ)
──── = Σ_u ──── · ───── · ──── · ──────
∂ξ       ∂Î(u)   ∂μ'     ∂(T·μ)   ∂ξ
```

**SE(3) Jacobian:**

Cho p = T · μ (transformed point):

```
∂p      ∂(R·μ + t)
──── = ────────────
∂ξ         ∂ξ

     = [I₃  |  -(R·μ)^]
       
       6x3 matrix
```

**Ví dụ số:**

```
R·μ = (1, 2, 3)

        ┌ 1 0 0 |  0  3 -2 ┐
∂p/∂ξ = | 0 1 0 | -3  0  1 |
        └ 0 0 1 |  2 -1  0 ┘
```

---

# PHẦN 3: GAUSSIAN MAP OPTIMIZATION (MAPPING)

## 3.1 Mục tiêu

Cho:
- Set of keyframes K = {(I_t, D_t, T_t)}
- Current Gaussian parameters Θ

Optimize Θ để minimize reconstruction loss.

## 3.2 Gaussian Parameters

```
Θ = {μ_i, q_i, s_i, α_i, sh_i}_{i=1}^{N}

μ_i ∈ ℝ³         // Position
q_i ∈ ℝ⁴         // Quaternion (rotation)
s_i ∈ ℝ³         // Scale (log-space)
α_i ∈ ℝ          // Opacity (before sigmoid)
sh_i ∈ ℝ^{48}    // SH coefficients (16 per RGB channel)
```

## 3.3 Loss Function

```
L_map = Σ_{t∈K} [ L_photo^t + λ_d·L_depth^t ] + λ_reg·L_reg
```

**Regularization Terms:**

```
L_reg = L_opacity + L_scale + L_dist
```

**Opacity Regularization:**

```
L_opacity = (1/N) · Σ_i min(α_i, 1-α_i)
```

→ Khuyến khích α gần 0 hoặc 1 (tránh semi-transparent)

**Scale Regularization:**

```
L_scale = (1/N) · Σ_i max(0, ‖s_i‖ - s_max)
```

→ Penalize Gaussians quá lớn

**Distance Regularization (optional):**

```
L_dist = (1/N) · Σ_i min_{j≠i} ‖μ_i - μ_j‖²
```

→ Khuyến khích Gaussians spread out

## 3.4 Adaptive Density Control

**Densification (Clone/Split):**

Khi gradient lớn → cần thêm Gaussians:

```
if ‖∇_μ L‖ > τ_grad:
    if ‖s‖ > τ_scale:
        # Split: Chia Gaussian thành 2
        G_new1, G_new2 = split(G)
    else:
        # Clone: Copy Gaussian và shift
        G_new = clone(G, offset=∇_μ direction)
```

**Split Operation:**

```
Original:          After Split:
    ╭───╮              ╭─╮ ╭─╮
   /     \            /   X   \
  |   ●   |    →     | ●     ● |
   \     /            \   X   /
    ╰───╯              ╰─╯ ╰─╯
    
New scale = old_scale / 1.6
New positions = μ ± s · principal_axis
```

**Clone Operation:**

```
Original:          After Clone:
    ╭───╮              ╭───╮ ╭───╮
   /     \            /     \     \
  |   ●   |    →     |   ●   ●   |
   \     /            \     /     /
    ╰───╯              ╰───╯ ╰───╯
    
New position = μ + small_offset
```

**Pruning:**

Xóa Gaussians không useful:

```
if α_i < τ_α:           # Too transparent
    remove(G_i)
    
if ‖s_i‖ > τ_large:     # Too large
    remove(G_i)
    
if ‖s_i‖ < τ_small:     # Too small
    remove(G_i)
```

## 3.5 Optimization Schedule

```python
def optimize_map(G, keyframes, num_iters=30000):
    optimizer = Adam(G.parameters(), lr=0.001)
    
    for iter in range(num_iters):
        # Sample random keyframe
        I, D, T = random_choice(keyframes)
        
        # Render
        I_hat, D_hat = render(G, T)
        
        # Compute loss
        loss = compute_loss(I, I_hat, D, D_hat)
        
        # Backprop and update
        loss.backward()
        optimizer.step()
        
        # Density control (every N iterations)
        if iter % 100 == 0:
            densify_and_prune(G)
        
        # Reset opacity (every M iterations)
        if iter % 3000 == 0:
            reset_opacity(G)  # Set all α to low value
```

---

# PHẦN 4: SUBMAP MANAGEMENT

## 4.1 Tại sao cần Submaps?

**Vấn đề với single global map:**
- Memory explosion cho large scenes
- Optimization khó converge
- Khó parallel processing
- Loop closure cần re-optimize toàn bộ map

**Giải pháp: Submap-based Architecture**

```
Scene divided into submaps:

    Submap 1        Submap 2        Submap 3
   ┌────────┐      ┌────────┐      ┌────────┐
   | G_1^1  |      | G_2^1  |      | G_3^1  |
   | G_1^2  |      | G_2^2  |      | G_3^2  |
   |  ...   |      |  ...   |      |  ...   |
   | G_1^N1 |      | G_2^N2 |      | G_3^N3 |
   └────────┘      └────────┘      └────────┘
       |               |               |
       v               v               v
    T_1^ref         T_2^ref         T_3^ref
```

## 4.2 Submap Structure

```
Submap M_k = {
    G_k = {G_i}         // Set of Gaussians (local frame)
    T_k^ref             // Reference pose (world frame)
    K_k = {(I_t, D_t, T_t)}  // Keyframes
    bounds              // Bounding box
}
```

## 4.3 Submap Creation Criteria

Tạo submap mới khi một trong các điều kiện sau:

**1. Translation threshold:**

```
‖t_current - t_ref‖ > τ_trans  (e.g., 2 meters)
```

**2. Rotation threshold:**

```
angle(R_current, R_ref) > τ_rot  (e.g., 30°)

where angle = arccos((tr(R_ref^T · R_current) - 1) / 2)
```

**3. Frame count threshold:**

```
|K_k| > N_max  (e.g., 200 frames)
```

**4. Coverage-based:**

```
overlap(current_view, submap_bounds) < τ_overlap
```

## 4.4 Submap Creation Process

```python
def maybe_create_new_submap(current_frame, current_submap):
    I, D, T = current_frame
    T_ref = current_submap.T_ref
    
    # Check translation
    translation = norm(T.translation - T_ref.translation)
    if translation > TAU_TRANS:
        return create_new_submap(current_frame)
    
    # Check rotation
    R_rel = T_ref.rotation.T @ T.rotation
    angle = arccos((trace(R_rel) - 1) / 2)
    if angle > TAU_ROT:
        return create_new_submap(current_frame)
    
    # Check frame count
    if len(current_submap.keyframes) > N_MAX:
        return create_new_submap(current_frame)
    
    return None  # Continue with current submap


def create_new_submap(frame):
    I, D, T = frame
    
    new_submap = Submap()
    new_submap.T_ref = T
    new_submap.keyframes = [frame]
    new_submap.gaussians = initialize_gaussians(I, D, T)
    
    return new_submap
```

## 4.5 Local vs Global Coordinates

**Local Frame:** Gaussians stored relative to submap reference pose

```
μ_local = T_ref^{-1} · μ_world
```

**Global Frame:** For visualization and loop closure

```
μ_world = T_ref · μ_local
```

**Transformation between frames:**

```
Submap k với T_k^ref:

    World Frame              Local Frame (Submap k)
         |                         |
         |   T_k^ref               |   I (identity)
         v                         v
      Origin                    Submap origin
         |                         |
         |   T_t (camera)          |   T_t^local = T_k^ref^{-1} · T_t
         v                         v
      Camera                    Camera (local)
```

## 4.6 Submap Merging (Optional)

Khi hai submaps overlap nhiều, có thể merge:

```python
def merge_submaps(M_i, M_j, T_ij):
    """
    Merge submap j into submap i.
    """
    # Transform all Gaussians from j to i's frame
    for G in M_j.gaussians:
        G.position = T_ij @ G.position
        G.rotation = T_ij.rotation @ G.rotation
    
    # Add to submap i
    M_i.gaussians.extend(M_j.gaussians)
    M_i.keyframes.extend(M_j.keyframes)
    
    # Remove redundant Gaussians
    prune_duplicates(M_i)
    
    return M_i
```

---

# PHẦN 5: LOOP CLOSURE DETECTION

## 5.1 Two-Stage Detection Pipeline

```
Stage 1: Visual Place Recognition (Coarse)
         |
         | Candidate loops
         v
Stage 2: Geometric Verification (Fine)
         |
         | Verified loops
         v
    Loop Closure
```

## 5.2 Stage 1: Visual Place Recognition

**Global Image Descriptor:**

Dùng learned descriptors như NetVLAD, DBoW2:

```
f_t = Encoder(I_t) ∈ ℝ^d  (e.g., d=256)
```

**NetVLAD Architecture:**

```
Input Image I_t
      |
      v
   CNN Backbone (VGG, ResNet)
      |
      v
   Local Features F ∈ ℝ^{H×W×D}
      |
      v
   VLAD Pooling
      |
      v
   Global Descriptor f ∈ ℝ^{K×D}
      |
      v
   L2 Normalize
```

**VLAD Pooling:**

```
V_k = Σ_i a_k(f_i) · (f_i - c_k)

where:
- c_k: k-th cluster center
- a_k(f_i): soft assignment of f_i to cluster k
- V_k: residual accumulation for cluster k
```

**Similarity Search:**

```
s(t, t') = f_t · f_t' / (‖f_t‖ · ‖f_t'‖)  # Cosine similarity
```

**Loop Candidate Selection:**

```python
def find_loop_candidates(f_t, database, tau_sim=0.8, tau_temporal=50):
    candidates = []
    
    for t', f_t' in database:
        # Skip recent frames (temporal constraint)
        if abs(t - t') < tau_temporal:
            continue
        
        # Compute similarity
        sim = cosine_similarity(f_t, f_t')
        
        if sim > tau_sim:
            candidates.append((t', sim))
    
    # Return top-k candidates
    return sorted(candidates, key=lambda x: -x[1])[:k]
```

## 5.3 Stage 2: Geometric Verification

**Mục đích:** Filter false positives từ visual matching

**Method 1: Feature Matching + RANSAC**

```python
def geometric_verify_features(I_i, I_j):
    # Extract local features
    kp_i, desc_i = extract_features(I_i)  # e.g., SuperPoint
    kp_j, desc_j = extract_features(I_j)
    
    # Match features
    matches = match(desc_i, desc_j)  # e.g., SuperGlue
    
    # RANSAC for essential matrix
    E, inliers = cv2.findEssentialMat(
        kp_i[matches], kp_j[matches], 
        method=cv2.RANSAC, threshold=1.0
    )
    
    # Check inlier ratio
    inlier_ratio = sum(inliers) / len(matches)
    
    return inlier_ratio > 0.3
```

**Method 2: Depth-based Overlap (LoopSplat's approach)**

```python
def geometric_verify_depth(G_i, G_j, T_query):
    """
    Verify loop by checking depth overlap.
    """
    # Render depth from both submaps
    D_i = render_depth(G_i, T_query)
    D_j = render_depth(G_j, T_query)
    
    # Count consistent pixels
    valid_i = D_i > 0
    valid_j = D_j > 0
    consistent = abs(D_i - D_j) < tau_depth
    
    overlap = sum(valid_i & valid_j & consistent) / sum(valid_i | valid_j)
    
    return overlap > 0.3
```

**Visualization:**

```
Depth from Submap i:          Depth from Submap j:
┌──────────────────┐          ┌──────────────────┐
| ▓▓▓░░░░░░▓▓▓▓▓▓ |          | ▓▓▓░░░░░░▓▓▓▓▓▓ |
| ▓▓▓▓░░░░▓▓▓▓▓▓▓ |          | ▓▓▓▓░░░░▓▓▓▓▓▓▓ |
| ▓▓▓▓▓░░▓▓▓▓▓▓▓▓ |    vs    | ▓▓▓▓▓░░▓▓▓▓▓▓▓▓ |
| ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ |          | ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ |
└──────────────────┘          └──────────────────┘

If depths are similar → Verified loop!
```

## 5.4 Loop Detection Algorithm

```python
class LoopDetector:
    def __init__(self):
        self.descriptor_database = {}
        self.submap_database = {}
    
    def add_keyframe(self, t, I, submap_id):
        # Compute and store descriptor
        f = self.encoder(I)
        self.descriptor_database[t] = (f, submap_id)
    
    def detect_loop(self, t, I, current_submap):
        f_t = self.encoder(I)
        
        # Stage 1: Visual search
        candidates = []
        for t', (f_t', submap_id) in self.descriptor_database.items():
            # Skip same submap
            if submap_id == current_submap.id:
                continue
            
            sim = cosine_similarity(f_t, f_t')
            if sim > self.tau_sim:
                candidates.append((t', submap_id, sim))
        
        # Stage 2: Geometric verification
        for t', submap_id, sim in sorted(candidates, key=lambda x: -x[2]):
            other_submap = self.submap_database[submap_id]
            
            if self.geometric_verify(current_submap, other_submap):
                return Loop(current_submap.id, submap_id, t, t')
        
        return None
```

---

# PHẦN 6: 3DGS REGISTRATION (CHI TIẾT)

## 6.1 Problem Setup

```
Given:
- Submap i: G_i = {G_i^1, G_i^2, ..., G_i^{N_i}}
- Submap j: G_j = {G_j^1, G_j^2, ..., G_j^{N_j}}
- Loop detection at frames (t_i, t_j)

Find:
- Relative transformation T_ij ∈ SE(3)
  such that: T_j^world = T_ij · T_i^world
```

## 6.2 Why Not Just Use ICP?

**Traditional ICP Pipeline:**

```
G_i, G_j (3DGS)
    |
    v Extract points
P_i, P_j (Point clouds)
    |
    v ICP
T_ij
```

**Problems:**
1. **Information loss:** 3DGS → point cloud loses color, opacity, shape
2. **Slow:** ICP needs nearest neighbor search (O(N log N) per iteration)
3. **Not differentiable:** Can't integrate into end-to-end training

## 6.3 LoopSplat's Approach

```
G_i, G_j (3DGS)
    |
    v Render at multiple viewpoints
Images/Depths
    |
    v Differentiable optimization
T_ij
```

**Advantages:**
1. **Full information:** Uses color AND geometry
2. **Fast:** GPU rasterization
3. **Differentiable:** End-to-end trainable

## 6.4 Multi-View Registration Loss

**Viewpoint Selection:**

```python
def select_viewpoints(G_i, G_j, T_i_ref, T_j_ref, K=5):
    """
    Select K diverse viewpoints in overlapping region.
    """
    # Get keyframe poses from both submaps
    poses_i = [kf.pose for kf in G_i.keyframes]
    poses_j = [kf.pose for kf in G_j.keyframes]
    
    # Find overlapping region
    overlap_poses = []
    for T in poses_i + poses_j:
        if is_in_overlap(T, G_i, G_j):
            overlap_poses.append(T)
    
    # Cluster to get diverse viewpoints
    viewpoints = kmeans_cluster(overlap_poses, K)
    
    return viewpoints
```

**Loss Computation:**

```python
def registration_loss(G_i, G_j, T_ij, viewpoints):
    total_loss = 0
    
    for T_k in viewpoints:
        # Render from submap i
        I_i, D_i = render(G_i, T_k)
        
        # Render from submap j (with transformation)
        T_jk = T_ij @ T_k
        I_j, D_j = render(G_j, T_jk)
        
        # Photometric loss
        L_photo = mean(abs(I_i - I_j))
        
        # Depth loss
        valid_mask = (D_i > 0) & (D_j > 0)
        L_depth = mean(abs(D_i[valid_mask] - D_j[valid_mask]))
        
        total_loss += L_photo + lambda_d * L_depth
    
    return total_loss / len(viewpoints)
```

## 6.5 Detailed Gradient Derivation

**Setup:**

```
L = L(I_i, I_j(T_ij))

where I_j(T_ij) = Render(G_j, T_ij · T_k)
```

**Chain Rule:**

```
∂L       ∂L      ∂I_j     ∂μ'_j    ∂μ_j^cam   ∂(T_ij·μ_j)
──── = ────── · ────── · ────── · ──────── · ───────────
∂ξ      ∂I_j    ∂μ'_j   ∂μ_j^cam  ∂(T_ij·μ_j)     ∂ξ
```

**Term 1: ∂L/∂I_j**

For L1 loss:

```
∂L/∂I_j = sign(I_i - I_j)
```

**Term 2: ∂I_j/∂μ'_j (Rasterizer Gradient)**

From alpha blending:

```
I_j(u) = Σ_k c_k · α_k · G_k(u) · T_k

∂I_j(u)         ∂G_k(u)
─────── = Σ_k c_k · α_k · ─────── · T_k
 ∂μ'_j           ∂μ'_j

where:
∂G_k(u)
─────── = G_k(u) · Σ'^{-1} · (u - μ'_k)
 ∂μ'_k
```

**Term 3: ∂μ'_j/∂μ_j^cam (Projection Jacobian)**

```
        ┌ f_x/z    0      -f_x·x/z² ┐
∂μ'/∂μ = |                          |
        └   0    f_y/z    -f_y·y/z² ┘
```

**Term 4: ∂μ_j^cam/∂(T_ij·μ_j)**

Camera transform:

```
μ_j^cam = T_k · (T_ij · μ_j)

∂μ_j^cam/∂(T_ij·μ_j) = R_k  (rotation part of T_k)
```

**Term 5: ∂(T_ij·μ_j)/∂ξ (SE(3) Jacobian)**

```
T_ij · μ_j = R_ij · μ_j + t_ij

∂(T_ij·μ_j)
─────────── = [I₃  |  -(R_ij·μ_j)^]
    ∂ξ

           ┌ 1 0 0 |   0       (Rμ)_z  -(Rμ)_y ┐
         = | 0 1 0 | -(Rμ)_z    0       (Rμ)_x |
           └ 0 0 1 |  (Rμ)_y  -(Rμ)_x    0     ┘
```

## 6.6 Complete Optimization Algorithm

```python
def register_3dgs(G_i, G_j, viewpoints, max_iters=100, lr=0.01):
    """
    Register two 3DGS submaps.
    """
    # Step 1: Initialize with feature matching
    T_ij = initialize_ransac(G_i, G_j, viewpoints[0])
    
    # Step 2: Convert to Lie algebra
    xi = SE3_to_se3(T_ij)  # ℝ⁶
    xi.requires_grad = True
    
    optimizer = torch.optim.Adam([xi], lr=lr)
    
    for iter in range(max_iters):
        optimizer.zero_grad()
        
        # Convert back to SE(3)
        T_ij = se3_to_SE3(xi)
        
        # Compute loss
        loss = registration_loss(G_i, G_j, T_ij, viewpoints)
        
        # Backprop
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([xi], max_norm=1.0)
        
        # Update
        optimizer.step()
        
        if loss < threshold:
            break
    
    return se3_to_SE3(xi)
```

## 6.7 Initialization Strategies

**Strategy 1: Identity (if submaps are close)**

```
T_ij^init = I
```

**Strategy 2: Relative from Reference Poses**

```
T_ij^init = T_j^ref · (T_i^ref)^{-1}
```

**Strategy 3: Feature Matching + RANSAC**

```python
def initialize_ransac(G_i, G_j, T_view):
    # Render images
    I_i = render_color(G_i, T_view)
    I_j = render_color(G_j, T_view)
    
    # Extract and match features
    kp_i, desc_i = superpoint(I_i)
    kp_j, desc_j = superpoint(I_j)
    matches = superglue(desc_i, desc_j)
    
    # Get 3D points from depth
    D_i = render_depth(G_i, T_view)
    D_j = render_depth(G_j, T_view)
    
    pts_3d_i = backproject(kp_i[matches], D_i, T_view)
    pts_3d_j = backproject(kp_j[matches], D_j, T_view)
    
    # RANSAC for rigid transformation
    T_ij, inliers = ransac_rigid(pts_3d_i, pts_3d_j)
    
    return T_ij
```

## 6.8 Handling Partial Overlap

**Problem:** Submaps may only partially overlap

**Solution:** Masked loss computation

```python
def masked_registration_loss(G_i, G_j, T_ij, viewpoints):
    total_loss = 0
    total_weight = 0
    
    for T_k in viewpoints:
        I_i, D_i = render(G_i, T_k)
        I_j, D_j = render(G_j, T_ij @ T_k)
        
        # Create overlap mask
        valid_i = D_i > 0
        valid_j = D_j > 0
        overlap_mask = valid_i & valid_j
        
        if overlap_mask.sum() < min_overlap:
            continue
        
        # Compute loss only on overlapping pixels
        L_photo = mean(abs(I_i[overlap_mask] - I_j[overlap_mask]))
        L_depth = mean(abs(D_i[overlap_mask] - D_j[overlap_mask]))
        
        weight = overlap_mask.sum()
        total_loss += (L_photo + lambda_d * L_depth) * weight
        total_weight += weight
    
    return total_loss / total_weight
```

---

# PHẦN 7: POSE GRAPH OPTIMIZATION

## 7.1 Pose Graph Structure

**Nodes:** Submap reference poses

```
V = {T_1, T_2, ..., T_M}  where T_k ∈ SE(3)
```

**Edges:** Constraints between submaps

```
E = E_odom ∪ E_loop

E_odom = {(k, k+1) : consecutive submaps}
E_loop = {(i, j) : loop closures}
```

**Visualization:**

```
    T_1 ────────── T_2 ────────── T_3 ────────── T_4
     |              |              |              |
     |  Odometry    |   Odometry   |   Odometry   |
     |    edge      |     edge     |     edge     |
     └──────────────┴──────────────┴──────────────┘
                    |
                    | Loop edge (detected)
                    |
     T_1 ───────────┴───────────── T_4
```

## 7.2 Edge Constraints

**Odometry Edge (k, k+1):**

```
T̂_{k,k+1} = T_k^{-1} · T_{k+1}  (measured from tracking)

Ω_{k,k+1} = Information matrix (inverse covariance)
```

**Loop Edge (i, j):**

```
T̂_{i,j} = result from 3DGS registration

Ω_{i,j} = estimated from registration Hessian
```

## 7.3 Error Function

**Relative Pose Error:**

```
e_{ij}(T_i, T_j) = Log(T̂_{ij}^{-1} · T_i^{-1} · T_j)
```

**Logarithm Map (SE(3) → se(3)):**

```
Log(T) = Log([R|t]) = [log(R); V^{-1}·t]

where log(R) = θ · (R - R^T) / (2 sin θ)
      θ = arccos((tr(R) - 1) / 2)
```

**Chi-squared Error:**

```
χ²_{ij} = e_{ij}^T · Ω_{ij} · e_{ij}
```

## 7.4 Full Objective Function

```
L_PGO = Σ_{(i,j)∈E} ρ(e_{ij}^T · Ω_{ij} · e_{ij})
```

**Robust Kernels:**

```
Huber:
         ┌ s                    if s < δ²
ρ(s) =   |
         └ 2δ√s - δ²            otherwise

Cauchy:
ρ(s) = δ² · log(1 + s/δ²)

Tukey:
         ┌ (δ²/6)·[1-(1-s/δ²)³]   if s < δ²
ρ(s) =   |
         └ δ²/6                    otherwise
```

**Why Robust Kernels?**
- Outlier loop closures có thể corrupt optimization
- Robust kernel giảm weight của large errors

## 7.5 Gauss-Newton Optimization

**Linearization:**

```
e_{ij}(T_i ⊕ Δξ_i, T_j ⊕ Δξ_j) ≈ e_{ij} + J_i·Δξ_i + J_j·Δξ_j
```

**Jacobians:**

```
       ∂e_{ij}              ∂e_{ij}
J_i = ──────── ,     J_j = ────────
        ∂ξ_i                 ∂ξ_j
```

For SE(3):

```
J_i = -Ad_{T_j^{-1}·T_i}    (Adjoint representation)
J_j = I_{6×6}
```

**Adjoint of SE(3):**

```
           ┌ R    0 ┐
Ad_T =     |        |
           └ t^·R  R┘
```

## 7.6 Normal Equations

**Linearized Problem:**

```
min_{Δξ} Σ_{(i,j)} ‖e_{ij} + J_i·Δξ_i + J_j·Δξ_j‖²_{Ω_{ij}}
```

**Normal Equations:**

```
H · Δξ = -b
```

**Hessian Matrix H:**

```
H = Σ_{(i,j)} J_{ij}^T · Ω_{ij} · J_{ij}

where J_{ij} = [... J_i ... J_j ...]  (sparse)
```

**Structure of H:**

```
         node 1   node 2   node 3   node 4
       ┌─────────────────────────────────────┐
node 1 |  H_11    H_12                       |  ← edge (1,2)
       |                                     |
node 2 |  H_21    H_22    H_23               |  ← edges (1,2), (2,3)
       |                                     |
node 3 |          H_32    H_33    H_34       |  ← edges (2,3), (3,4)
       |                                     |
node 4 |                  H_43    H_44 + ... |  ← edge (3,4) + loop
       └─────────────────────────────────────┘
                                        |
                                   Loop edge (1,4)
                                   adds to H_11, H_14, H_41, H_44
```

**Information Vector b:**

```
b = Σ_{(i,j)} J_{ij}^T · Ω_{ij} · e_{ij}
```

## 7.7 Sparse Solver

H is sparse → use sparse Cholesky decomposition:

```
H = L · L^T    (Cholesky)

Solve: L · y = -b
       L^T · Δξ = y
```

**Libraries:**
- g2o
- GTSAM
- Ceres Solver

## 7.8 Iterative Algorithm

```python
def pose_graph_optimization(nodes, edges, max_iters=100):
    """
    Optimize pose graph using Gauss-Newton.
    """
    for iter in range(max_iters):
        # Build system
        H = sparse_matrix(6*len(nodes), 6*len(nodes))
        b = zeros(6*len(nodes))
        
        total_chi2 = 0
        
        for (i, j), T_ij_measured, Omega_ij in edges:
            # Compute error
            T_i, T_j = nodes[i], nodes[j]
            e_ij = compute_error(T_i, T_j, T_ij_measured)
            
            # Compute Jacobians
            J_i, J_j = compute_jacobians(T_i, T_j)
            
            # Apply robust kernel
            chi2 = e_ij.T @ Omega_ij @ e_ij
            weight = robust_weight(chi2)  # ∂ρ/∂s
            
            # Add to system
            H[i*6:(i+1)*6, i*6:(i+1)*6] += weight * J_i.T @ Omega_ij @ J_i
            H[i*6:(i+1)*6, j*6:(j+1)*6] += weight * J_i.T @ Omega_ij @ J_j
            H[j*6:(j+1)*6, i*6:(i+1)*6] += weight * J_j.T @ Omega_ij @ J_i
            H[j*6:(j+1)*6, j*6:(j+1)*6] += weight * J_j.T @ Omega_ij @ J_j
            
            b[i*6:(i+1)*6] += weight * J_i.T @ Omega_ij @ e_ij
            b[j*6:(j+1)*6] += weight * J_j.T @ Omega_ij @ e_ij
            
            total_chi2 += chi2
        
        # Fix first node (gauge freedom)
        H[0:6, 0:6] += 1e6 * eye(6)
        
        # Solve
        delta_xi = sparse_solve(H, -b)
        
        # Update nodes
        for i in range(len(nodes)):
            delta_T = exp_se3(delta_xi[i*6:(i+1)*6])
            nodes[i] = delta_T @ nodes[i]
        
        # Check convergence
        if norm(delta_xi) < 1e-6:
            break
    
    return nodes
```

## 7.9 Information Matrix Estimation

**From Registration Hessian:**

```
Ω_{ij} ≈ (∂²L_reg / ∂ξ²)^{-1}
```

**Approximation using Jacobian:**

```
Ω_{ij} = J_{reg}^T · J_{reg}

where J_{reg} = ∂r / ∂ξ  (residual Jacobian)
```

**Empirical:**

```
Ω_{odom} = diag([100, 100, 100, 1000, 1000, 1000])  // [trans, rot]
Ω_{loop} = diag([50, 50, 50, 500, 500, 500])        // Less certain
```

---

# PHẦN 8: SUBMAP ALIGNMENT

## 8.1 After PGO

Sau khi PGO optimize các node poses, cần update Gaussians:

```
Before PGO:          After PGO:
T_k^old              T_k^new
   |                    |
   v                    v
Submap k            Submap k (transformed)
```

## 8.2 Rigid Transformation of Gaussians

**Position Update:**

```
μ_i^new = T_k^new · (T_k^old)^{-1} · μ_i^old
        = ΔT_k · μ_i^old

where ΔT_k = T_k^new · (T_k^old)^{-1}
```

**Rotation Update (Quaternion):**

```
q_i^new = Δq_k ⊗ q_i^old

where Δq_k = quaternion(ΔR_k)
      ΔR_k = R_k^new · (R_k^old)^T
```

**Quaternion Multiplication:**

```
q_1 ⊗ q_2 = [w_1·w_2 - v_1·v_2,
             w_1·v_2 + w_2·v_1 + v_1 × v_2]

where q = [w, v] = [w, x, y, z]
```

**Covariance Update:**

```
Σ_i^new = ΔR_k · Σ_i^old · ΔR_k^T
```

## 8.3 Implementation

```python
def align_submap(submap, T_old, T_new):
    """
    Rigidly transform all Gaussians in submap.
    """
    # Compute delta transform
    delta_T = T_new @ inverse(T_old)
    delta_R = delta_T[:3, :3]
    delta_t = delta_T[:3, 3]
    delta_q = rotation_matrix_to_quaternion(delta_R)
    
    for G in submap.gaussians:
        # Update position
        G.position = delta_R @ G.position + delta_t
        
        # Update rotation (quaternion)
        G.rotation = quaternion_multiply(delta_q, G.rotation)
        
        # Update covariance
        # Note: Since Σ = R·S·S^T·R^T, and we're applying delta_R:
        # Σ_new = delta_R · R · S · S^T · R^T · delta_R^T
        #       = (delta_R · R) · S · S^T · (delta_R · R)^T
        # So we just need to update the rotation quaternion
        
    # Update keyframe poses
    for kf in submap.keyframes:
        kf.pose = delta_T @ kf.pose
    
    return submap
```

## 8.4 Handling Submap Boundaries

**Problem:** After alignment, adjacent submaps may have gaps or overlaps

**Solution 1: Boundary Blending**

```python
def blend_submaps(M_i, M_j, overlap_region):
    """
    Blend Gaussians in overlapping region.
    """
    for G_i in M_i.gaussians:
        if in_region(G_i.position, overlap_region):
            # Find nearest Gaussian in M_j
            G_j = find_nearest(G_i, M_j.gaussians)
            
            if distance(G_i, G_j) < threshold:
                # Merge Gaussians
                G_merged = merge_gaussians(G_i, G_j)
                
                # Remove originals, add merged
                M_i.remove(G_i)
                M_j.remove(G_j)
                M_i.add(G_merged)
```

**Solution 2: Joint Optimization**

After alignment, do a few iterations of joint optimization:

```python
def joint_optimization(submaps, num_iters=1000):
    # Merge all Gaussians
    all_gaussians = []
    for M in submaps:
        all_gaussians.extend(M.gaussians)
    
    # Merge all keyframes
    all_keyframes = []
    for M in submaps:
        all_keyframes.extend(M.keyframes)
    
    # Optimize jointly
    for iter in range(num_iters):
        kf = random_choice(all_keyframes)
        I_hat, D_hat = render(all_gaussians, kf.pose)
        
        loss = compute_loss(kf.image, I_hat, kf.depth, D_hat)
        loss.backward()
        
        optimizer.step()
```

---

# PHẦN 9: ONLINE VS OFFLINE LOOP CLOSURE

## 9.1 Offline Mode

```
Collect all frames
       |
       v
Build all submaps
       |
       v
Detect all loops
       |
       v
Single PGO
       |
       v
Align all submaps
       |
       v
Final map
```

**Pros:**
- Simpler implementation
- Can use global information

**Cons:**
- Drift accumulates until end
- Large errors may be hard to correct

## 9.2 Online Mode (LoopSplat)

```
For each frame:
    |
    ├──▶ Track & Map (current submap)
    |
    ├──▶ Check for loop closure
    |         |
    |         ├──▶ If loop found:
    |         |       |
    |         |       ├──▶ 3DGS Registration
    |         |       |
    |         |       ├──▶ Incremental PGO
    |         |       |
    |         |       └──▶ Align affected submaps
    |         |
    |         └──▶ Continue
    |
    └──▶ Maybe create new submap
```

**Pros:**
- Drift corrected continuously
- Smaller corrections (easier to optimize)
- Real-time capable

**Cons:**
- More complex implementation
- May miss some loops (not seeing future)

## 9.3 Incremental PGO

Instead of re-optimizing entire graph:

```python
def incremental_pgo(pose_graph, new_loop_edge):
    """
    Incrementally update pose graph with new loop.
    """
    # Add new edge
    pose_graph.add_edge(new_loop_edge)
    
    # Find affected nodes (connected component)
    affected = find_affected_nodes(pose_graph, new_loop_edge)
    
    # Only optimize affected portion
    optimize_subgraph(pose_graph, affected, num_iters=10)
    
    # Update submaps
    for node_id in affected:
        T_old = pose_graph.get_old_pose(node_id)
        T_new = pose_graph.get_pose(node_id)
        
        if not close_enough(T_old, T_new):
            align_submap(submaps[node_id], T_old, T_new)
```

## 9.4 Experimental Results (from paper)

**Replica Dataset (small rooms):**

```
                ATE [cm]
Method          Offline    Online
────────────────────────────────
Gaussian-SLAM   0.48       -
LoopSplat       0.45       0.44    (similar, few loops)
```

**ScanNet Dataset (large scenes):**

```
                ATE [cm]
Method          Offline    Online
────────────────────────────────
Gaussian-SLAM   8.2        -
LoopSplat       4.1        3.2     (significant improvement!)
```

**Insight:** Online LC crucial for large scenes with many loops.

---

# PHẦN 10: COMPLETE SYSTEM PIPELINE

## 10.1 Full Algorithm

```python
class LoopSplat:
    def __init__(self, config):
        self.config = config
        self.submaps = []
        self.current_submap = None
        self.pose_graph = PoseGraph()
        self.loop_detector = LoopDetector()
        
    def process_frame(self, rgb, depth, timestamp):
        """
        Process a single RGB-D frame.
        """
        # ═══════════════════════════════════════════
        # STEP 1: TRACKING
        # ═══════════════════════════════════════════
        if self.current_submap is None:
            # First frame - initialize
            T = np.eye(4)
            self.current_submap = self.create_submap(rgb, depth, T)
            self.pose_graph.add_node(T)
        else:
            # Track against current submap
            T_init = self.predict_pose()  # Motion model
            T = self.track(rgb, depth, self.current_submap, T_init)
        
        # ═══════════════════════════════════════════
        # STEP 2: MAPPING
        # ═══════════════════════════════════════════
        is_keyframe = self.is_keyframe(T)
        
        if is_keyframe:
            # Add keyframe to current submap
            self.current_submap.add_keyframe(rgb, depth, T)
            
            # Optimize Gaussian map
            self.optimize_map(self.current_submap)
            
            # Add to loop detector database
            self.loop_detector.add_keyframe(timestamp, rgb, 
                                            self.current_submap.id)
        
        # ═══════════════════════════════════════════
        # STEP 3: SUBMAP MANAGEMENT
        # ═══════════════════════════════════════════
        if self.should_create_submap(T):
            # Finalize current submap
            self.submaps.append(self.current_submap)
            
            # Add odometry edge to pose graph
            T_rel = self.current_submap.T_ref_inv @ T
            self.pose_graph.add_edge(
                self.current_submap.id,
                len(self.submaps),  # new submap id
                T_rel,
                self.config.odometry_info
            )
            
            # Create new submap
            self.current_submap = self.create_submap(rgb, depth, T)
            self.pose_graph.add_node(T)
        
        # ═══════════════════════════════════════════
        # STEP 4: LOOP CLOSURE
        # ═══════════════════════════════════════════
        if is_keyframe:
            loop = self.loop_detector.detect(timestamp, rgb, 
                                             self.current_submap)
            
            if loop is not None:
                # 3DGS Registration
                T_loop = self.register_submaps(
                    self.submaps[loop.submap_i],
                    self.submaps[loop.submap_j]
                )
                
                # Add loop edge
                info_matrix = self.estimate_info_matrix(T_loop)
                self.pose_graph.add_edge(
                    loop.submap_i,
                    loop.submap_j,
                    T_loop,
                    info_matrix
                )
                
                # Pose Graph Optimization
                self.pose_graph.optimize()
                
                # Align affected submaps
                self.align_submaps()
        
        return T
    
    # ═══════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════
    
    def track(self, rgb, depth, submap, T_init):
        T = T_init.copy()
        
        for iter in range(self.config.track_iters):
            I_hat, D_hat = render(submap.gaussians, T)
            
            loss = self.tracking_loss(rgb, I_hat, depth, D_hat)
            grad = compute_gradient(loss, T)
            
            T = update_se3(T, -self.config.track_lr * grad)
            
            if converged(loss):
                break
        
        return T
    
    def optimize_map(self, submap, num_iters=100):
        optimizer = Adam(submap.gaussians.parameters(), lr=0.001)
        
        for iter in range(num_iters):
            kf = random.choice(submap.keyframes)
            I_hat, D_hat = render(submap.gaussians, kf.pose)
            
            loss = self.mapping_loss(kf.rgb, I_hat, kf.depth, D_hat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iter % 50 == 0:
                self.densify_and_prune(submap.gaussians)
    
    def register_submaps(self, submap_i, submap_j):
        viewpoints = self.select_viewpoints(submap_i, submap_j)
        
        T_ij = self.initialize_registration(submap_i, submap_j)
        
        for iter in range(self.config.reg_iters):
            loss = self.registration_loss(
                submap_i.gaussians, 
                submap_j.gaussians,
                T_ij, 
                viewpoints
            )
            
            grad = compute_gradient(loss, T_ij)
            T_ij = update_se3(T_ij, -self.config.reg_lr * grad)
        
        return T_ij
    
    def align_submaps(self):
        for i, submap in enumerate(self.submaps):
            T_old = submap.T_ref
            T_new = self.pose_graph.get_pose(i)
            
            if not np.allclose(T_old, T_new):
                delta_T = T_new @ np.linalg.inv(T_old)
                self.transform_submap(submap, delta_T)
                submap.T_ref = T_new
    
    def get_global_map(self):
        """
        Merge all submaps into global map.
        """
        global_gaussians = []
        
        for submap in self.submaps:
            for G in submap.gaussians:
                G_world = self.transform_gaussian(G, submap.T_ref)
                global_gaussians.append(G_world)
        
        return global_gaussians
```

## 10.2 Configuration

```python
@dataclass
class LoopSplatConfig:
    # Tracking
    track_iters: int = 100
    track_lr: float = 0.01
    lambda_geo: float = 0.5
    
    # Mapping
    map_iters_per_frame: int = 100
    densify_interval: int = 50
    prune_opacity_threshold: float = 0.01
    
    # Submap
    trans_threshold: float = 2.0  # meters
    rot_threshold: float = 30.0   # degrees
    max_frames_per_submap: int = 200
    
    # Loop Detection
    similarity_threshold: float = 0.8
    temporal_threshold: int = 50
    overlap_threshold: float = 0.3
    
    # Registration
    reg_iters: int = 100
    reg_lr: float = 0.01
    num_viewpoints: int = 5
    
    # PGO
    pgo_iters: int = 100
    odometry_info: np.ndarray = np.diag([100, 100, 100, 1000, 1000, 1000])
```

---

# PHẦN 11: KẾT QUẢ VÀ SO SÁNH

## 11.1 Datasets

| Dataset | Type | # Scenes | Characteristics |
|---------|------|----------|-----------------|
| Replica | Synthetic | 8 | Clean, small rooms |
| TUM-RGBD | Real | 3 | Office/desk scenes |
| ScanNet | Real | 6 | Large indoor spaces |
| ScanNet++ | Real | 2 | High-quality scans |

## 11.2 Metrics

**Tracking: ATE (Absolute Trajectory Error)**

```
ATE = sqrt((1/N) · Σ_i ‖T_i^est - T_i^gt‖²)
```

**Mapping: Accuracy, Completion, Chamfer Distance**

```
Accuracy = mean(min_q d(p, q)) for p in reconstruction
Completion = mean(min_p d(p, q)) for q in ground truth
Chamfer = (Accuracy + Completion) / 2
```

**Rendering: PSNR, SSIM, LPIPS**

```
PSNR = 10 · log10(MAX² / MSE)
SSIM = structural similarity
LPIPS = learned perceptual similarity
```

## 11.3 Tracking Results (ATE [cm])

```
                    Replica    TUM-RGBD   ScanNet
─────────────────────────────────────────────────
NICE-SLAM           1.69       3.00       8.64
Point-SLAM          0.52       1.31       10.24
Gaussian-SLAM       0.48       1.10       8.23
SplaTAM             0.36       1.21       7.45
Loopy-SLAM          0.42       1.05       4.52
─────────────────────────────────────────────────
LoopSplat           0.44       0.89       3.21   ← Best on real data
```

## 11.4 Rendering Results (PSNR)

```
                    Replica    TUM-RGBD   ScanNet
─────────────────────────────────────────────────
Point-SLAM          32.1       21.5       19.2
Gaussian-SLAM       33.5       22.1       20.1
SplaTAM             34.2       23.8       21.5
Loopy-SLAM          33.8       22.5       20.8
─────────────────────────────────────────────────
LoopSplat           34.8       23.2       22.3   ← Best overall
```

## 11.5 Runtime Comparison

```
                    Track     Map      Loop Reg   Total
                   (ms/f)   (ms/f)      (ms)     (FPS)
────────────────────────────────────────────────────────
Loopy-SLAM          85       120        520        5
Gaussian-SLAM       45        80          -       10
LoopSplat           52        95        150        8
```

## 11.6 Memory Usage

```
                    Peak Memory (GB)
                    Replica  ScanNet
─────────────────────────────────────
Point-SLAM           8.2      15.3
Gaussian-SLAM        4.5       9.2
LoopSplat            5.1      10.5
```

---

# PHẦN 12: LIMITATIONS VÀ FUTURE WORK

## 12.1 Current Limitations

**1. RGB-D Only**
- Requires depth sensor
- Can't work with monocular camera

**2. Static Scene Assumption**
- Moving objects cause artifacts
- No dynamic object handling

**3. Computational Cost**
- Still not real-time on consumer hardware
- GPU memory intensive

**4. Loop Detection**
- Relies on visual similarity
- May miss loops in repetitive environments

## 12.2 Potential Improvements

**1. Monocular Extension**

```
Ideas:
- Use depth prediction network
- Bundle adjustment for depth
- Combine with visual odometry
```

**2. Dynamic Object Handling**

```
Ideas:
- Semantic segmentation to mask dynamics
- Separate dynamic/static Gaussians
- Temporal consistency checks
```

**3. Efficiency**

```
Ideas:
- Level-of-detail rendering
- Hierarchical submaps
- Mixed precision training
```

**4. Better Loop Detection**

```
Ideas:
- Semantic place recognition
- 3D descriptor matching
- Learning-based verification
```

## 12.3 Research Directions

1. **Multi-agent SLAM**: Multiple robots sharing 3DGS maps
2. **Long-term Mapping**: Handling changes over time
3. **Semantic Integration**: Object-level SLAM with 3DGS
4. **Physics Integration**: Gaussian dynamics for robotics

---

# PHẦN 13: TÓM TẮT

## Key Contributions

1. **3DGS Registration:** Phương pháp đầu tiên sử dụng differentiable rendering của 3DGS để tính loop edge constraints

2. **Submap-based Architecture:** Chia scene thành các submaps độc lập cho scalability

3. **Online Loop Closure:** Thực hiện loop closure và PGO online để liên tục sửa drift

4. **Unified Representation:** Sử dụng 3DGS cho tracking, mapping, và loop closure

## Pipeline Overview

```
Input: RGB-D Stream
         |
         v
┌─────────────────────────────────────────────────────────┐
|                      FRONT-END                          |
├─────────────────────────────────────────────────────────┤
|  ┌──────────────┐     ┌──────────────┐                  |
|  |   Tracking   |────▶|   Mapping    |                  |
|  | (Frame-to-   |     |  (Gaussian   |                  |
|  |   Model)     |     | Optimization)|                  |
|  └──────────────┘     └──────────────┘                  |
|         |                    |                          |
|         v                    v                          |
|  ┌─────────────────────────────────────┐                |
|  |        Submap Management            |                |
|  |  (Create new submap when threshold  |                |
|  |           exceeded)                 |                |
|  └─────────────────────────────────────┘                |
└─────────────────────────────────────────────────────────┘
                         |
                         v
┌─────────────────────────────────────────────────────────┐
|                      BACK-END                           |
├─────────────────────────────────────────────────────────┤
|  ┌──────────────┐     ┌──────────────┐                  |
|  |    Loop      |────▶|    3DGS      |                  |
|  |  Detection   |     | Registration |                  |
|  |  (Visual +   |     |  (Compute    |                  |
|  |  Geometric)  |     |  T_ij)       |                  |
|  └──────────────┘     └──────────────┘                  |
|                              |                          |
|                              v                          |
|                 ┌──────────────────────┐                |
|                 |   Pose Graph         |                |
|                 |   Optimization       |                |
|                 └──────────────────────┘                |
|                              |                          |
|                              v                          |
|                 ┌──────────────────────┐                |
|                 |  Submap Alignment    |                |
|                 └──────────────────────┘                |
└─────────────────────────────────────────────────────────┘
                         |
                         v
Output: Globally Consistent 3DGS Map + Camera Trajectory
```

---

# References

- **Original Paper:** [arXiv:2408.10154](https://arxiv.org/abs/2408.10154)
- **Project Page:** [https://loopsplat.github.io/](https://loopsplat.github.io/)
- **Code:** [https://github.com/GradientSpaces/LoopSplat](https://github.com/GradientSpaces/LoopSplat)

**Related Works:**
- 3D Gaussian Splatting: [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- Gaussian-SLAM: [https://github.com/VladimirYugay/Gaussian-SLAM](https://github.com/VladimirYugay/Gaussian-SLAM)
- SplaTAM: [https://spla-tam.github.io/](https://spla-tam.github.io/)
