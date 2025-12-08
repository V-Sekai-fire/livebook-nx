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
#   --output-format "usdc"       Output format: usdc only (default: "usdc")
#                                Binary USDC format preserves quads and embedded materials
#                                with optimal performance and file size
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
# All dependencies managed by uv (no pip)
Pythonx.uv_init("""
[project]
name = "unirig-generation"
version = "0.0.0"
requires-python = "==3.11.*"
dependencies = [
  "numpy<2.0",
  "bpy==4.*",
  "pillow",
  "opencv-python",
  "torch==2.7.0",
  "torchvision",
  "pytorch-lightning",
  "lightning",
  "huggingface-hub",
  "einops",
  "tqdm",
  "trimesh",
  "cumm-cu118",
  "spconv-cu118==2.3.8",
  "torch_scatter @ https://data.pyg.org/whl/torch-2.7.0%2Bcu118/torch_scatter-2.1.2%2Bpt27cu118-cp311-cp311-win_amd64.whl ; sys_platform == 'win32'",
  "torch_scatter @ https://data.pyg.org/whl/torch-2.7.0%2Bcu118/torch_scatter-2.1.2%2Bpt27cu118-cp311-cp311-linux_x86_64.whl ; sys_platform == 'linux'",
  "torch-cluster @ https://data.pyg.org/whl/torch-2.7.0%2Bcu118/torch_cluster-1.6.3%2Bpt27cu118-cp311-cp311-win_amd64.whl ; sys_platform == 'win32'",
  "torch-cluster @ https://data.pyg.org/whl/torch-2.7.0%2Bcu118/torch_cluster-1.6.3%2Bpt27cu118-cp311-cp311-linux_x86_64.whl ; sys_platform == 'linux'",
  "scipy",
  "pyyaml",
  "omegaconf",
  "hydra-core",
  "fvcore",
  "point-cloud-utils",
  "transformers==4.51.3",
  "python-box",
  "addict",
  "timm",
  "fast-simplification",
  "open3d",
  "pyrender",
  "wandb",
  # Note: flash-attn is installed separately after torch is available
  # to avoid build isolation issues. See Python code below.
  # Note: UniRig is used from local thirdparty/UniRig directory
]

[tool.uv.sources]
torch = { index = "pytorch-cu118" }
torchvision = { index = "pytorch-cu118" }

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true
""")

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
        --output-format, -f "usdc"    Output format: usdc only (default: "usdc")
                                       Binary USDC format preserves quads and embedded materials
                                       with optimal performance and file size
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
      output_format: Keyword.get(opts, :output_format, "usdc"),
      seed: Keyword.get(opts, :seed, 42),
      skeleton_only: skeleton_only,
      skin_only: skin_only,
      skeleton_task: Keyword.get(opts, :skeleton_task),
      skin_task: Keyword.get(opts, :skin_task)
    }

    # Validate output_format - only USDC allowed
    if config.output_format != "usdc" do
      IO.puts("Error: Only USDC output format is supported, got: #{config.output_format}")
      IO.puts("  Please use 'usdc' as output format")
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
import yaml
from box import Box
import lightning as L
from math import ceil

# Fix for PyTorch 2.7+ weights_only loading - allow Box objects in checkpoints
torch.serialization.add_safe_globals([Box])

# Helper function to load configs (from run.py)
def load_config(task: str, path: str) -> Box:
    if path.endswith('.yaml'):
        path = path.removesuffix('.yaml')
    path += '.yaml'
    print(f"load {task} config: {path}")
    return Box(yaml.safe_load(open(path, 'r')))

# Helper function to run UniRig inference (replicates run.py logic)
def run_unirig_inference(
    task_path: str,
    seed: int,
    input_path: str,
    output_path: str,
    npz_dir: str,
    data_name: str = None,
    unirig_base_path: str = None
):
    # Run UniRig inference directly without subprocess
    torch.set_float32_matmul_precision('high')
    L.seed_everything(seed, workers=True)

    # Load task config
    task = load_config('task', task_path)
    mode = task.mode
    assert mode in ['train', 'predict', 'validate']

    # Import UniRig modules
    from src.data.extract import get_files
    from src.data.datapath import Datapath
    from src.data.dataset import UniRigDatasetModule, DatasetConfig
    from src.data.transform import TransformConfig
    from src.tokenizer.spec import TokenizerConfig
    from src.tokenizer.parse import get_tokenizer
    from src.model.parse import get_model
    from src.system.parse import get_system, get_writer
    from src.inference.download import download
    from lightning.pytorch.callbacks import ModelCheckpoint

    # Load configs first (needed for extraction)
    data_config = load_config('data', os.path.join('configs/data', task.components.data))
    transform_config = load_config('transform', os.path.join('configs/transform', task.components.transform))

    # Get files
    files = get_files(
        data_name=task.components.data_name,
        inputs=input_path,
        input_dataset_dir=None,
        output_dataset_dir=npz_dir,
        force_override=True,
        warning=False,
    )

    # Extract mesh files to create raw_data.npz if they don't exist
    from src.data.extract import extract_builtin
    import time
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")

    # Check which files need extraction
    data_name_actual = task.components.get('data_name', 'raw_data.npz')
    files_to_extract = []
    for input_file, output_dir in files:
        raw_data_npz = os.path.join(output_dir, data_name_actual)
        if not os.path.exists(raw_data_npz):
            files_to_extract.append((input_file, output_dir))

    if files_to_extract:
        print(f"\\n=== Extracting {len(files_to_extract)} mesh file(s) ===")
        # Get target_count from data config if available, default to 50000
        target_count = data_config.get('faces_target_count', 50000)
        extract_builtin(
            output_folder=npz_dir,
            target_count=target_count,
            num_runs=1,
            id=0,
            time=timestamp,
            files=files_to_extract,
        )
        print("✓ Mesh extraction complete")

    files = [f[1] for f in files]
    datapath = Datapath(files=files, cls=None)

    # Get tokenizer
    tokenizer_config = task.components.get('tokenizer', None)
    if tokenizer_config is not None:
        tokenizer_config = load_config('tokenizer', os.path.join('configs/tokenizer', task.components.tokenizer))
        from src.tokenizer.spec import TokenizerConfig
        tokenizer_config = TokenizerConfig.parse(config=tokenizer_config)

    # Get data name
    data_name_actual = task.components.get('data_name', 'raw_data.npz')
    if data_name is not None:
        data_name_actual = data_name

    # Get predict dataset and transform
    predict_dataset_config = data_config.get('predict_dataset_config', None)
    if predict_dataset_config is not None:
        predict_dataset_config = DatasetConfig.parse(config=predict_dataset_config).split_by_cls()

    predict_transform_config = transform_config.get('predict_transform_config', None)
    if predict_transform_config is not None:
        predict_transform_config = TransformConfig.parse(config=predict_transform_config)

    # Get model
    model_config = task.components.get('model', None)
    if model_config is not None:
        model_config = load_config('model', os.path.join('configs/model', model_config))
        if tokenizer_config is not None:
            tokenizer = get_tokenizer(config=tokenizer_config)
        else:
            tokenizer = None
        model = get_model(tokenizer=tokenizer, **model_config)
    else:
        model = None

    # Set up data module
    data = UniRigDatasetModule(
        process_fn=None if model is None else model._process_fn,
        predict_dataset_config=predict_dataset_config,
        predict_transform_config=predict_transform_config,
        tokenizer_config=tokenizer_config,
        debug=False,
        data_name=data_name_actual,
        datapath=datapath,
        cls=None,
    )

    # Get writer callback
    writer_config = task.get('writer', None)
    callbacks = []
    if writer_config is not None:
        assert predict_transform_config is not None, 'missing predict_transform_config in transform'
        writer_config['npz_dir'] = npz_dir
        # For skeleton generation, set output_dir to npz_dir so files are saved there
        # For skin generation, we can use None (default behavior)
        if writer_config.get('export_npz') == 'predict_skeleton':
            writer_config['output_dir'] = npz_dir  # Save skeleton npz in npz_dir
            writer_config['user_mode'] = False  # Need to save npz for skin generation
        else:
            writer_config['output_dir'] = None
            writer_config['user_mode'] = True
        writer_config['output_name'] = output_path
        callbacks.append(get_writer(**writer_config, order_config=predict_transform_config.order_config))

    # Get system
    system_config = task.components.get('system', None)
    if system_config is not None:
        system_config = load_config('system', os.path.join('configs/system', system_config))
        optimizer_config = task.get('optimizer', None)
        loss_config = task.get('loss', None)
        scheduler_config = task.get('scheduler', None)

        train_dataset_config = data_config.get('train_dataset_config', None)
        if train_dataset_config is not None:
            train_dataset_config = DatasetConfig.parse(config=train_dataset_config)

        system = get_system(
            **system_config,
            model=model,
            optimizer_config=optimizer_config,
            loss_config=loss_config,
            scheduler_config=scheduler_config,
            steps_per_epoch=1 if train_dataset_config is None else
            ceil(len(data.train_dataloader()) // 1 // 1),
        )
    else:
        system = None

    # Get trainer
    trainer_config = task.get('trainer', {})

    # Set checkpoint path
    resume_from_checkpoint = task.get('resume_from_checkpoint', None)
    resume_from_checkpoint = download(resume_from_checkpoint)

    # Fix for PyTorch 2.7+ weights_only loading - allow Box objects in checkpoints
    # This needs to be done before creating the trainer
    try:
        torch.serialization.add_safe_globals([Box])
    except Exception:
        # If already added, ignore
        pass

    trainer = L.Trainer(
        callbacks=callbacks,
        logger=None,
        **trainer_config,
    )

    # Run prediction
    assert resume_from_checkpoint is not None, 'expect resume_from_checkpoint in task'
    trainer.predict(system, datamodule=data, ckpt_path=resume_from_checkpoint, return_predictions=False)

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
original_cwd = os.getcwd()

# Create output directory (absolute path)
output_dir = Path(original_cwd) / "output"
output_dir.mkdir(exist_ok=True, parents=True)

# Create timestamped folder for this run (matching other generation scripts)
import time
tag = time.strftime("%Y%m%d_%H_%M_%S")
export_dir = output_dir / tag
export_dir.mkdir(exist_ok=True, parents=True)

# Create intermediate directory for skeleton/skin files within the timestamped folder
intermediate_dir = export_dir / "intermediate"
intermediate_dir.mkdir(exist_ok=True, parents=True)

print("\\n=== Step 1: Setup UniRig Environment ===")

# Check if UniRig repository is available
# Try to find UniRig in thirdparty relative to current working directory
unirig_path = Path.cwd() / "thirdparty" / "UniRig"

if not unirig_path.exists():
    unirig_path = None
    print("⚠ UniRig repository not found in thirdparty/UniRig")
    print(f"  Searched at: {Path.cwd() / 'thirdparty' / 'UniRig'}")
    print("  Please ensure thirdparty/UniRig exists")
    raise FileNotFoundError("UniRig repository not found. Expected at: thirdparty/UniRig")
else:
    unirig_path = unirig_path.resolve()
    print(f"✓ Using UniRig from: {unirig_path}")

# Add UniRig to Python path if found locally
if unirig_path and unirig_path.exists():
    if str(unirig_path) not in sys.path:
        sys.path.insert(0, str(unirig_path))

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Setup npz directory for intermediate files (absolute path)
# Store NPZ files in the intermediate directory within the timestamped folder
npz_dir = intermediate_dir
npz_dir.mkdir(exist_ok=True, parents=True)

# Change to UniRig directory if available for relative paths in configs
if unirig_path and unirig_path.exists():
    os.chdir(str(unirig_path))

# Determine workflow
if skin_only:
    print("\\n=== Step 2: Generate Skinning Weights ===")
    print("⚠ Note: Skin-only mode requires an existing skeleton file")
    print("  Please provide skeleton file path or use full pipeline")

    skeleton_path = export_dir / "skeleton.fbx"
    if not skeleton_path.exists():
        print(f"✗ Error: Skeleton file not found: {skeleton_path}")
        print("  Run without --skin-only to generate skeleton first")
        raise FileNotFoundError(f"Skeleton file not found: {skeleton_path}")

    # Generate skinning weights using direct function call
    print(f"Generating skinning weights for: {mesh_path}")
    print(f"Using skeleton: {skeleton_path}")

    skin_output = export_dir / "skin.fbx"
    skin_task_path = skin_task or "configs/task/quick_inference_unirig_skin.yaml"

    try:
        run_unirig_inference(
            task_path=skin_task_path,
            seed=seed,
            input_path=str(skeleton_path),
            output_path=str(skin_output),
            npz_dir=str(npz_dir),
            data_name="predict_skeleton.npz",
            unirig_base_path=str(unirig_path) if unirig_path else None
        )
        print(f"✓ Skinning weights generated: {skin_output}")

        # Merge skeleton and skin
        final_output = export_dir / f"rigged.{output_format}"
        from src.inference.merge import transfer
        transfer(
            source=str(skin_output),
            target=str(Path(mesh_path).resolve()),
            output=str(final_output),
            add_root=False
        )
        print(f"✓ Rigged model saved: {final_output}")
    except Exception as e:
        print(f"✗ Error during skin generation: {e}")
        import traceback
        traceback.print_exc()
        raise

elif skeleton_only:
    print("\\n=== Step 2: Generate Skeleton ===")
    print(f"Generating skeleton for: {mesh_path}")

    skeleton_output = export_dir / f"skeleton.{output_format}"
    skeleton_task_path = skeleton_task or "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"

    try:
        run_unirig_inference(
            task_path=skeleton_task_path,
            seed=seed,
            input_path=str(Path(mesh_path).resolve()),
            output_path=str(skeleton_output),
            npz_dir=str(npz_dir),
            data_name=None,
            unirig_base_path=str(unirig_path) if unirig_path else None
        )
        print(f"✓ Skeleton generated: {skeleton_output}")
    except Exception as e:
        print(f"✗ Error during skeleton generation: {e}")
        import traceback
        traceback.print_exc()
        raise

else:
    # Full pipeline: skeleton + skin + merge
    print("\\n=== Step 2: Generate Skeleton ===")
    print(f"Generating skeleton for: {mesh_path}")

    skeleton_output = export_dir / "skeleton.fbx"
    skeleton_task_path = skeleton_task or "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"

    try:
        run_unirig_inference(
            task_path=skeleton_task_path,
            seed=seed,
            input_path=str(Path(mesh_path).resolve()),
            output_path=str(skeleton_output),
            npz_dir=str(npz_dir),
            data_name=None,
            unirig_base_path=str(unirig_path) if unirig_path else None
        )
        print(f"✓ Skeleton generated: {skeleton_output}")
    except Exception as e:
        print(f"✗ Error during skeleton generation: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("\\n=== Step 3: Generate Skinning Weights ===")
    print(f"Generating skinning weights for skeleton: {skeleton_output}")

    # Verify predict_skeleton.npz exists before skin generation
    # The skeleton generation saves it relative to the input file location
    # We need to find where it was actually saved
    mesh_stem = Path(mesh_path).stem
    mesh_dir = Path(mesh_path).parent

    # Check multiple possible locations
    possible_paths = [
        Path(npz_dir) / mesh_stem / "predict_skeleton.npz",  # Expected location in npz_dir
        mesh_dir / mesh_stem / "predict_skeleton.npz",  # Where it might have been saved (relative to input)
        mesh_dir / "predict_skeleton.npz",  # Direct in input directory
    ]

    skeleton_npz_path = None
    for path in possible_paths:
        if path.exists():
            skeleton_npz_path = path
            print(f"Found predict_skeleton.npz at: {skeleton_npz_path}")
            # Copy it to the expected location if it's not there
            expected_path = Path(npz_dir) / mesh_stem / "predict_skeleton.npz"
            if skeleton_npz_path != expected_path:
                expected_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(skeleton_npz_path, expected_path)
                print(f"Copied predict_skeleton.npz to expected location: {expected_path}")
            break

    if skeleton_npz_path is None:
        # Last resort: search recursively
        import glob
        found_files = list(Path(npz_dir).rglob("predict_skeleton.npz"))
        found_files.extend(list(mesh_dir.rglob("predict_skeleton.npz")))
        if found_files:
            skeleton_npz_path = found_files[0]
            print(f"Found predict_skeleton.npz at: {skeleton_npz_path}")
            # Copy to expected location
            expected_path = Path(npz_dir) / mesh_stem / "predict_skeleton.npz"
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(skeleton_npz_path, expected_path)
            print(f"Copied predict_skeleton.npz to expected location: {expected_path}")
        else:
            raise FileNotFoundError(f"predict_skeleton.npz not found. Searched in: {possible_paths}")

    skin_output = export_dir / "skin.fbx"
    skin_task_path = skin_task or "configs/task/quick_inference_unirig_skin.yaml"

    try:
        # Use the original mesh path, not skeleton_output, to maintain directory structure
        run_unirig_inference(
            task_path=skin_task_path,
            seed=seed,
            input_path=str(Path(mesh_path).resolve()),  # Use original mesh path, not skeleton file
            output_path=str(skin_output),
            npz_dir=str(npz_dir),
            data_name="predict_skeleton.npz",
            unirig_base_path=str(unirig_path) if unirig_path else None
        )
        print(f"✓ Skinning weights generated: {skin_output}")
    except Exception as e:
        print(f"✗ Error during skin generation: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("\\n=== Step 4: Merge Skeleton and Skin ===")
    print("Merging skeleton and skinning weights...")

    final_output = export_dir / f"rigged.{output_format}"

    try:
        from src.inference.merge import transfer
        transfer(
            source=str(skin_output),
            target=str(Path(mesh_path).resolve()),
            output=str(final_output),
            add_root=False
        )
        print(f"✓ Rigged model saved: {final_output}")
    except Exception as e:
        print(f"✗ Error during merge: {e}")
        import traceback
        traceback.print_exc()
        raise

# Restore original working directory
os.chdir(original_cwd)

print("\\n=== Complete ===")
print("3D model rigging completed successfully!")
print(f"\\nOutput files in {export_dir.name}/:")
if skeleton_only:
    print(f"  - {export_dir.name}/skeleton.{output_format} (Skeleton only)")
elif skin_only:
    print(f"  - {export_dir.name}/rigged.{output_format} (Rigged model)")
else:
    print(f"  - {export_dir.name}/skeleton.fbx (Skeleton)")
    print(f"  - {export_dir.name}/skin.fbx (Skinning weights)")
    print(f"  - {export_dir.name}/rigged.{output_format} (Final rigged model)")
    print(f"  - {export_dir.name}/intermediate/ (Intermediate NPZ files)")
""", %{})

IO.puts("\n=== Complete ===")
IO.puts("3D model rigging completed successfully!")
