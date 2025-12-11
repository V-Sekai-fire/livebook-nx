"""
Toy problem to test texture optimization assumptions.
This creates a simple mesh with known ground truth texture and tests if optimization can recover it.
"""
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

def create_simple_mesh():
    """Create a simple quad mesh (2 triangles forming a square)"""
    # Vertices of a unit square
    vertices = np.array([
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [1, 1, 0],  # 2
        [0, 1, 0],  # 3
    ], dtype=np.float32)
    
    # Two triangles forming a square
    faces = np.array([
        [0, 1, 2],  # Triangle 1
        [0, 2, 3],  # Triangle 2
    ], dtype=np.int32)
    
    # Simple UV coordinates mapping the square to texture space
    uvs = np.array([
        [0, 0],  # Vertex 0 -> UV (0, 0)
        [1, 0],  # Vertex 1 -> UV (1, 0)
        [1, 1],  # Vertex 2 -> UV (1, 1)
        [0, 1],  # Vertex 3 -> UV (0, 1)
    ], dtype=np.float32)
    
    return vertices, faces, uvs

def create_ground_truth_texture(texture_size=64):
    """Create a simple ground truth texture (gradient from red to blue)"""
    texture = np.zeros((texture_size, texture_size, 3), dtype=np.float32)
    for i in range(texture_size):
        for j in range(texture_size):
            # Horizontal gradient: red on left, blue on right
            texture[i, j, 0] = j / texture_size  # Red channel
            texture[i, j, 1] = 0.5  # Green channel
            texture[i, j, 2] = 1.0 - j / texture_size  # Blue channel
    return texture

def render_texture_with_grid_sample(texture, uv_map, height=64, width=64):
    """
    Render texture using grid_sample (same as in optimization).
    
    Args:
        texture: (1, H, W, 3) tensor in [0, 1] range
        uv_map: (H, W, 2) tensor in [0, 1] range
        height, width: Output resolution
    
    Returns:
        rendered: (H, W, 3) tensor
    """
    # Ensure texture is (1, C, H, W) for grid_sample
    texture_for_sampling = texture.permute(0, 3, 1, 2)  # (1, 3, H, W)
    
    # Convert UV from [0, 1] to [-1, 1] and flip Y
    uv_normalized = torch.stack([
        uv_map[..., 0] * 2.0 - 1.0,
        (1.0 - uv_map[..., 1]) * 2.0 - 1.0
    ], dim=-1)
    uv_grid = uv_normalized.unsqueeze(0)  # (1, H, W, 2)
    
    # Sample texture
    rendered = F.grid_sample(
        texture_for_sampling, uv_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    rendered = rendered.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
    rendered = torch.clamp(rendered, 0.0, 1.0)
    
    return rendered

def create_observation_from_texture(texture, uv_map, height=64, width=64):
    """Create an observation by rendering the texture"""
    texture_tensor = torch.tensor(texture).float().cuda().unsqueeze(0)  # (1, H, W, 3)
    uv_map_tensor = torch.tensor(uv_map).float().cuda()  # (H, W, 2)
    observation = render_texture_with_grid_sample(texture_tensor, uv_map_tensor, height, width)
    return observation

def test_texture_optimization():
    """Test if we can recover ground truth texture through optimization"""
    print("=== Texture Optimization Toy Problem ===")
    
    # Parameters
    texture_size = 64
    height, width = 64, 64
    
    # Create ground truth
    print("1. Creating ground truth texture...")
    gt_texture = create_ground_truth_texture(texture_size)
    gt_texture_tensor = torch.tensor(gt_texture).float().cuda().unsqueeze(0)  # (1, H, W, 3)
    
    # Create simple UV map (identity mapping for this test)
    print("2. Creating UV map...")
    uv_map = np.zeros((height, width, 2), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            uv_map[i, j, 0] = j / width  # U coordinate
            uv_map[i, j, 1] = i / height  # V coordinate
    
    # Create observation from ground truth
    print("3. Creating observation from ground truth...")
    observation = create_observation_from_texture(gt_texture, uv_map, height, width)
    mask = torch.ones((height, width), dtype=torch.bool, device='cuda')
    
    print(f"   Observation shape: {observation.shape}")
    print(f"   Observation range: [{observation.min():.3f}, {observation.max():.3f}]")
    print(f"   Observation mean: {observation.mean():.3f}")
    
    # Initialize texture to optimize
    print("4. Initializing texture to optimize...")
    # Test 1: Initialize with zeros
    texture_zero = torch.nn.Parameter(torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32).cuda())
    # Test 2: Initialize with small random values
    texture_rand = torch.nn.Parameter(torch.randn((1, texture_size, texture_size, 3), dtype=torch.float32).cuda() * 0.01)
    
    # Test both initializations
    for init_name, texture in [("zeros", texture_zero), ("random", texture_rand)]:
        print(f"\n--- Testing with {init_name} initialization ---")
        texture_param = torch.nn.Parameter(texture.data.clone())
        optimizer = torch.optim.Adam([texture_param], lr=1e-2)
        
        losses = []
        total_steps = 500
        
        for step in tqdm(range(total_steps), desc=f"Optimizing ({init_name})"):
            optimizer.zero_grad()
            
            # Render current texture
            render = render_texture_with_grid_sample(texture_param, 
                                                     torch.tensor(uv_map).float().cuda(),
                                                     height, width)
            
            # Compute loss
            loss = F.l1_loss(render[mask], observation[mask])
            loss.backward()
            optimizer.step()
            
            # Clamp texture to [0, 1]
            with torch.no_grad():
                texture_param.data = torch.clamp(texture_param.data, 0.0, 1.0)
            
            losses.append(loss.item())
            
            if step % 100 == 0:
                # Compute error vs ground truth
                gt_error = F.l1_loss(texture_param.data, gt_texture_tensor)
                print(f"   Step {step}: loss={loss.item():.4f}, gt_error={gt_error.item():.4f}")
        
        print(f"   Final loss: {losses[-1]:.4f}")
        print(f"   Loss change: {losses[0]:.4f} -> {losses[-1]:.4f} (improvement: {losses[0] - losses[-1]:.4f})")
        print(f"   Final GT error: {F.l1_loss(texture_param.data, gt_texture_tensor).item():.4f}")
        
        # Check if loss is decreasing
        if losses[-1] < losses[0] * 0.9:  # At least 10% improvement
            print(f"   ✓ Loss is decreasing ({init_name} initialization works)")
        else:
            print(f"   ✗ Loss is NOT decreasing significantly ({init_name} initialization may have issues)")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_texture_optimization()

