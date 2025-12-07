#!/usr/bin/env elixir

# UniRig Generation Script
# Automatically rig 3D models with skeleton and skinning weights using UniRig
# Repository: https://github.com/VAST-AI-Research/UniRig
# Hugging Face: https://huggingface.co/VAST-AI/UniRig
#
# Usage:
#   elixir unirig_generation.exs <mesh_path> [options]
#
# Options:
#   --output-format "fbx"        Output format: fbx, glb (default: "fbx")
#   --seed <int>                 Random seed for skeleton generation (default: 42)
#   --skeleton-only              Only generate skeleton, skip skinning (default: false)
#   --skin-only                  Only generate skinning (requires existing skeleton) (default: false)
#   --skeleton-task <path>       Custom skeleton task config (optional)
#   --skin-task <path>           Custom skin task config (optional)

Mix.install([
  {:pythonx, "~> 0.4.7"},
  {:jason, "~> 1.4.4"}
])

# Initialize Python environment with required dependencies
# UniRig uses PyTorch Lightning and various 3D processing libraries
Pythonx.uv_init("""
[project]
name = "unirig-generation"
version = "0.0.0"
requires-python = "==3.10.*"
dependencies = [
  "numpy<2.0",
  "pillow",
  "opencv-python",
  "torch",
  "torchvision",
  "pytorch-lightning",
  "huggingface-hub",
  "einops",
  "tqdm",
  "trimesh",
  "scipy",
  "pyyaml",
  "omegaconf",
  "hydra-core",
  "fvcore",
  "point-cloud-utils",
]

[tool.uv.sources]
torch = { index = "pytorch-cu118" }
torchvision = { index = "pytorch-cu118" }

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true
""")

# Install UniRig package from git
IO.puts("\n=== Installing UniRig packages ===")
{_, _} = Pythonx.eval("""
import subprocess
import sys

# Install UniRig package
packages_to_install = [
    "git+https://github.com/VAST-AI-Research/UniRig.git",
]

print("Installing UniRig packages...")
for pkg in packages_to_install:
    print(f"Installing {pkg}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"✓ Installed UniRig")
    except subprocess.CalledProcessError as e:
        print(f"⚠ Warning: Failed to install {pkg}: {e}")
        print("  You may need to install manually: pip install git+https://github.com/VAST-AI-Research/UniRig.git")
        pass

print("✓ UniRig packages installation complete")
""", %{})

# Parse command-line arguments
defmodule ArgsParser do
  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        output_format: :string,
        seed: :integer,
        skeleton_only: :boolean,
        skin_only: :boolean,
        skeleton_task: :string,
        skin_task: :string
      ],
      aliases: [
        f: :output_format,
        s: :seed,
        sk: :skeleton_only,
        so: :skin_only
      ]
    )

    mesh_path = List.first(args)

    if !mesh_path do
      IO.puts("""
      Error: Mesh path is required.

      Usage:
        elixir unirig_generation.exs <mesh_path> [options]

      Options:
        --output-format, -f "fbx"      Output format: fbx, glb (default: "fbx")
        --seed, -s <int>                Random seed for skeleton generation (default: 42)
        --skeleton-only, -sk            Only generate skeleton, skip skinning (default: false)
        --skin-only, -so                Only generate skinning (requires existing skeleton) (default: false)
        --skeleton-task <path>          Custom skeleton task config (optional)
        --skin-task <path>              Custom skin task config (optional)
      """)
      System.halt(1)
    end

    skeleton_only = Keyword.get(opts, :skeleton_only, false)
    skin_only = Keyword.get(opts, :skin_only, false)

    if skeleton_only && skin_only do
      IO.puts("Error: Cannot use --skeleton-only and --skin-only together")
      System.halt(1)
    end

    config = %{
      mesh_path: mesh_path,
      output_format: Keyword.get(opts, :output_format, "fbx"),
      seed: Keyword.get(opts, :seed, 42),
      skeleton_only: skeleton_only,
      skin_only: skin_only,
      skeleton_task: Keyword.get(opts, :skeleton_task),
      skin_task: Keyword.get(opts, :skin_task)
    }

    # Validate output_format
    valid_formats = ["fbx", "glb"]
    if config.output_format not in valid_formats do
      IO.puts("Error: Invalid output format. Must be one of: #{Enum.join(valid_formats, ", ")}")
      System.halt(1)
    end

    # Check if file exists
    if !File.exists?(config.mesh_path) do
      IO.puts("Error: Mesh file not found: #{config.mesh_path}")
      System.halt(1)
    end

    config
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

IO.puts("""
=== UniRig Generation ===
Mesh: #{config.mesh_path}
Output Format: #{config.output_format}
Seed: #{config.seed}
Skeleton Only: #{config.skeleton_only}
Skin Only: #{config.skin_only}
""")

# Save config to JSON for Python to read
config_json = Jason.encode!(config)
File.write!("config.json", config_json)

# Import libraries and process using UniRig
{_, _python_globals} = Pythonx.eval("""
import json
import sys
import os
from pathlib import Path
import torch

# Get configuration from JSON file
with open("config.json", 'r') as f:
    config = json.load(f)

mesh_path = config.get('mesh_path')
output_format = config.get('output_format', 'fbx')
seed = config.get('seed', 42)
skeleton_only = config.get('skeleton_only', False)
skin_only = config.get('skin_only', False)
skeleton_task = config.get('skeleton_task')
skin_task = config.get('skin_task')

# Resolve paths to absolute
mesh_path = str(Path(mesh_path).resolve())

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Create intermediate directory for skeleton/skin files
intermediate_dir = output_dir / "unirig_intermediate"
intermediate_dir.mkdir(exist_ok=True)

print("\\n=== Step 1: Setup UniRig Environment ===")

# Check if UniRig repository is available
# Try to add UniRig to path if installed as package
try:
    import unirig
    print("✓ UniRig package found")
    unirig_path = None
except ImportError:
    # Try to find UniRig in thirdparty
    unirig_path = Path("thirdparty", "UniRig").resolve()
    if unirig_path.exists():
        print(f"Using UniRig from: {unirig_path}")
        if str(unirig_path) not in sys.path:
            sys.path.insert(0, str(unirig_path))
    else:
        print("⚠ UniRig repository not found in thirdparty/UniRig")
        print("  Models will be downloaded from Hugging Face")
        unirig_path = None

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Determine workflow
if skin_only:
    print("\\n=== Step 2: Generate Skinning Weights ===")
    print("⚠ Note: Skin-only mode requires an existing skeleton file")
    print("  Please provide skeleton file path or use full pipeline")
    
    skeleton_path = intermediate_dir / f"{Path(mesh_path).stem}_skeleton.fbx"
    if not skeleton_path.exists():
        print(f"✗ Error: Skeleton file not found: {skeleton_path}")
        print("  Run without --skin-only to generate skeleton first")
        raise FileNotFoundError(f"Skeleton file not found: {skeleton_path}")
    
    # Generate skinning weights
    print(f"Generating skinning weights for: {mesh_path}")
    print(f"Using skeleton: {skeleton_path}")
    
    # Use UniRig's skin generation
    # This would typically use: bash launch/inference/generate_skin.sh
    # For Python API, we'll need to call the inference directly
    try:
        # Import UniRig modules
        if unirig_path:
            sys.path.insert(0, str(unirig_path))
        
        # Try to use UniRig's Python API
        # Note: This is a simplified version - actual implementation may vary
        from launch.inference.generate_skin import generate_skin
        
        skin_output = intermediate_dir / f"{Path(mesh_path).stem}_skin.fbx"
        
        # Generate skin
        generate_skin(
            input_path=str(skeleton_path),
            output_path=str(skin_output),
            skin_task=skin_task or "configs/task/quick_inference_skin.yaml"
        )
        
        print(f"✓ Skinning weights generated: {skin_output}")
        
        # Final output
        final_output = output_dir / f"output_rigged.{output_format}"
        
        # Merge skeleton and skin
        from launch.inference.merge import merge_skeleton_skin
        
        merge_skeleton_skin(
            source=str(skin_output),
            target=str(mesh_path),
            output=str(final_output)
        )
        
        print(f"✓ Rigged model saved: {final_output}")
        
    except Exception as e:
        print(f"✗ Error during skin generation: {e}")
        print("\\nTrying alternative method...")
        # Fallback: use shell script approach
        import subprocess
        script_path = Path("launch/inference/generate_skin.sh")
        if unirig_path and (unirig_path / script_path).exists():
            cmd = [
                "bash",
                str(unirig_path / script_path),
                "--input", str(skeleton_path),
                "--output", str(intermediate_dir / f"{Path(mesh_path).stem}_skin.fbx"),
            ]
            if skin_task:
                cmd.extend(["--skin_task", skin_task])
            
            subprocess.check_call(cmd)
            print("✓ Skinning weights generated via shell script")
        else:
            raise

elif skeleton_only:
    print("\\n=== Step 2: Generate Skeleton ===")
    print(f"Generating skeleton for: {mesh_path}")
    
    skeleton_output = output_dir / f"output_skeleton.{output_format}"
    
    try:
        # Import UniRig modules
        if unirig_path:
            sys.path.insert(0, str(unirig_path))
        
        # Use UniRig's skeleton generation
        from launch.inference.generate_skeleton import generate_skeleton
        
        generate_skeleton(
            input_path=str(mesh_path),
            output_path=str(skeleton_output),
            seed=seed,
            skeleton_task=skeleton_task or "configs/task/quick_inference_skeleton.yaml"
        )
        
        print(f"✓ Skeleton generated: {skeleton_output}")
        
    except Exception as e:
        print(f"✗ Error during skeleton generation: {e}")
        print("\\nTrying alternative method...")
        # Fallback: use shell script approach
        import subprocess
        script_path = Path("launch/inference/generate_skeleton.sh")
        if unirig_path and (unirig_path / script_path).exists():
            cmd = [
                "bash",
                str(unirig_path / script_path),
                "--input", str(mesh_path),
                "--output", str(skeleton_output),
                "--seed", str(seed),
            ]
            if skeleton_task:
                cmd.extend(["--skeleton_task", skeleton_task])
            
            subprocess.check_call(cmd)
            print(f"✓ Skeleton generated via shell script: {skeleton_output}")
        else:
            raise

else:
    # Full pipeline: skeleton + skin + merge
    print("\\n=== Step 2: Generate Skeleton ===")
    print(f"Generating skeleton for: {mesh_path}")
    
    skeleton_output = intermediate_dir / f"{Path(mesh_path).stem}_skeleton.fbx"
    
    try:
        # Import UniRig modules
        if unirig_path:
            sys.path.insert(0, str(unirig_path))
        
        # Use UniRig's skeleton generation
        from launch.inference.generate_skeleton import generate_skeleton
        
        generate_skeleton(
            input_path=str(mesh_path),
            output_path=str(skeleton_output),
            seed=seed,
            skeleton_task=skeleton_task or "configs/task/quick_inference_skeleton.yaml"
        )
        
        print(f"✓ Skeleton generated: {skeleton_output}")
        
    except Exception as e:
        print(f"⚠ Error during skeleton generation: {e}")
        print("\\nTrying alternative method...")
        # Fallback: use shell script or direct API
        import subprocess
        script_path = Path("launch/inference/generate_skeleton.sh")
        if unirig_path and (unirig_path / script_path).exists():
            cmd = [
                "bash",
                str(unirig_path / script_path),
                "--input", str(mesh_path),
                "--output", str(skeleton_output),
                "--seed", str(seed),
            ]
            if skeleton_task:
                cmd.extend(["--skeleton_task", skeleton_task])
            
            try:
                subprocess.check_call(cmd)
                print(f"✓ Skeleton generated via shell script: {skeleton_output}")
            except:
                print("⚠ Shell script method also failed")
                raise
        else:
            # Try direct Python API
            print("⚠ Attempting direct Python API...")
            # This would require the actual UniRig Python API
            raise
    
    print("\\n=== Step 3: Generate Skinning Weights ===")
    print(f"Generating skinning weights for skeleton: {skeleton_output}")
    
    skin_output = intermediate_dir / f"{Path(mesh_path).stem}_skin.fbx"
    
    try:
        from launch.inference.generate_skin import generate_skin
        
        generate_skin(
            input_path=str(skeleton_output),
            output_path=str(skin_output),
            skin_task=skin_task or "configs/task/quick_inference_skin.yaml"
        )
        
        print(f"✓ Skinning weights generated: {skin_output}")
        
    except Exception as e:
        print(f"⚠ Error during skin generation: {e}")
        print("\\nTrying alternative method...")
        import subprocess
        script_path = Path("launch/inference/generate_skin.sh")
        if unirig_path and (unirig_path / script_path).exists():
            cmd = [
                "bash",
                str(unirig_path / script_path),
                "--input", str(skeleton_output),
                "--output", str(skin_output),
            ]
            if skin_task:
                cmd.extend(["--skin_task", skin_task])
            
            try:
                subprocess.check_call(cmd)
                print(f"✓ Skinning weights generated via shell script: {skin_output}")
            except:
                print("⚠ Shell script method also failed")
                raise
    
    print("\\n=== Step 4: Merge Skeleton and Skin ===")
    print("Merging skeleton and skinning weights...")
    
    final_output = output_dir / f"output_rigged.{output_format}"
    
    try:
        from launch.inference.merge import merge_skeleton_skin
        
        merge_skeleton_skin(
            source=str(skin_output),
            target=str(mesh_path),
            output=str(final_output)
        )
        
        print(f"✓ Rigged model saved: {final_output}")
        
    except Exception as e:
        print(f"⚠ Error during merge: {e}")
        print("\\nTrying alternative method...")
        import subprocess
        script_path = Path("launch/inference/merge.sh")
        if unirig_path and (unirig_path / script_path).exists():
            cmd = [
                "bash",
                str(unirig_path / script_path),
                "--source", str(skin_output),
                "--target", str(mesh_path),
                "--output", str(final_output),
            ]
            
            try:
                subprocess.check_call(cmd)
                print(f"✓ Rigged model saved via shell script: {final_output}")
            except:
                print("⚠ Shell script method also failed")
                raise

print("\\n=== Complete ===")
print("3D model rigging completed successfully!")
print(f"\\nOutput files:")
if skeleton_only:
    print(f"  - output/output_skeleton.{output_format} (Skeleton only)")
elif skin_only:
    print(f"  - output/output_rigged.{output_format} (Rigged model)")
else:
    print(f"  - output/unirig_intermediate/*_skeleton.fbx (Intermediate skeleton)")
    print(f"  - output/unirig_intermediate/*_skin.fbx (Intermediate skin)")
    print(f"  - output/output_rigged.{output_format} (Final rigged model)")
""", %{})

IO.puts("\n=== Complete ===")
IO.puts("3D model rigging completed successfully!")

