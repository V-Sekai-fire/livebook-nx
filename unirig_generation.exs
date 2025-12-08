#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
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
  "bpy==4.5.*",
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
  "torch",
  "libigl",
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
  def show_help do
    IO.puts("""
    UniRig Generation Script
    Automatically rig 3D models with skeleton and skinning weights using UniRig
    Repository: https://github.com/VAST-AI-Research/UniRig
    Hugging Face: https://huggingface.co/VAST-AI/UniRig

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
      --help, -h                       Show this help message

    Example:
      elixir unirig_generation.exs model.obj --seed 42
      elixir unirig_generation.exs model.usdc --skeleton-only
      elixir unirig_generation.exs model.glb --skin-only
    """)
  end

  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        output_format: :string,
        seed: :integer,
        skeleton_only: :boolean,
        skin_only: :boolean,
        skeleton_task: :string,
        skin_task: :string,
        help: :boolean
      ],
      aliases: [
        f: :output_format,
        s: :seed,
        sk: :skeleton_only,
        so: :skin_only,
        h: :help
      ]
    )

    if Keyword.get(opts, :help, false) do
      show_help()
      System.halt(0)
    end

    mesh_path = List.first(args)

    if !mesh_path do
      IO.puts("""
      Error: Mesh path is required.

      Usage:
        elixir unirig_generation.exs <mesh_path> [options]

      Use --help or -h for more information.
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
import warnings
from pathlib import Path
import torch
import yaml
from box import Box
import lightning as L
from math import ceil

# Suppress verbose warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*flash_attn.*')
warnings.filterwarnings('ignore', message='.*flash-attn.*')
warnings.filterwarnings('ignore', message='.*flash attention.*')
warnings.filterwarnings('ignore', message='.*BatchNorm.*')
# Suppress warnings from transformers and other libraries
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Create filtered stdout/stderr wrapper to suppress specific warning messages
class FilteredOutput:
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.filtered_patterns = [
            'flash_attn is disabled',
            'flash-attn is disabled',
            'flash attention is disabled',
            'use BatchNorm in ptv3obj',
            'WARNING: use BatchNorm',
            'Warning: flash_attn is disabled',
        ]
        self.buffer = ''

    def write(self, text):
        if not text:
            return

        # Handle partial lines (text might not end with newline)
        self.buffer += text
        if '\\n' in self.buffer:
            lines = self.buffer.split('\\n')
            # Keep the last incomplete line in buffer
            self.buffer = lines[-1]
            complete_lines = lines[:-1]

            # Filter complete lines
            filtered_lines = []
            for line in complete_lines:
                should_filter = False
                line_lower = line.lower()
                for pattern in self.filtered_patterns:
                    if pattern.lower() in line_lower:
                        should_filter = True
                        break
                if not should_filter:
                    filtered_lines.append(line)

            # Write filtered lines
            if filtered_lines:
                self.original_stream.write('\\n'.join(filtered_lines) + '\\n')

    def flush(self):
        # Flush any remaining buffer content (if it doesn't match filter)
        if self.buffer:
            should_filter = False
            buffer_lower = self.buffer.lower()
            for pattern in self.filtered_patterns:
                if pattern.lower() in buffer_lower:
                    should_filter = True
                    break
            if not should_filter:
                self.original_stream.write(self.buffer)
            self.buffer = ''
        self.original_stream.flush()

# Replace stdout and stderr with filtered versions
sys.stdout = FilteredOutput(sys.stdout)
sys.stderr = FilteredOutput(sys.stderr)

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
    # Patch UniRig's Exporter to use USD instead of FBX when path is USD
    # This removes all internal FBX usage in UniRig
    from src.data.exporter import Exporter
    original_export_fbx = Exporter._export_fbx

    def patched_export_fbx(self, path, vertices=None, joints=None, skin=None, parents=None, names=None, faces=None, extrude_size=0.03, group_per_vertex=-1, add_root=False, do_not_normalize=False, use_extrude_bone=True, use_connect_unique_child=True, extrude_from_parent=True, tails=None):
        # Always export as USD using Blender - no FBX usage
        # Convert .fbx path to .usdc if needed
        import bpy
        import os

        # Change file extension from .fbx to .usdc
        if path.lower().endswith('.fbx'):
            path = path[:-4] + '.usdc'
        elif not path.lower().endswith(('.usd', '.usda', '.usdc')):
            # If no extension or other extension, add .usdc
            path = path + '.usdc'

        # Build the armature/mesh in Blender (same as original _export_fbx)
        self._safe_make_dir(path)
        self._clean_bpy()
        self._make_armature(
            vertices=vertices,
            joints=joints,
            skin=skin,
            parents=parents,
            names=names,
            faces=faces,
            extrude_size=extrude_size,
            group_per_vertex=group_per_vertex,
            add_root=add_root,
            do_not_normalize=do_not_normalize,
            use_extrude_bone=use_extrude_bone,
            use_connect_unique_child=use_connect_unique_child,
            extrude_from_parent=extrude_from_parent,
            tails=tails,
        )

        # Export as USD - pipeline is now Blender/USD only
        bpy.ops.wm.usd_export(
            filepath=path,
            export_materials=True,
            export_textures=True,
            relative_paths=False,
            export_uvmaps=True,
            export_armatures=True,
            selected_objects_only=False,
            visible_objects_only=False,
            use_instancing=False,
            evaluation_mode='RENDER'
        )

    # Apply patch globally to remove all FBX usage
    Exporter._export_fbx = patched_export_fbx

    try:
        # Run UniRig inference directly without subprocess
        torch.set_float32_matmul_precision('high')
        L.seed_everything(seed, workers=True)

        # Load task config
        task = load_config('task', task_path)
        mode = task.mode
        assert mode in ['train', 'predict', 'validate']

        # Import UniRig modules (suppress warnings during import)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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

        # Get files - include USD formats in require_suffix
        # Extract just the filename stem to avoid nested directory structures
        # get_files creates output_dir based on the input path, which can create nested paths
        # We fix this by using just the filename stem for the output directory
        input_path_abs = os.path.abspath(input_path)
        input_basename = os.path.basename(input_path_abs)
        input_stem = os.path.splitext(input_basename)[0]  # Get filename without extension

        files = get_files(
            data_name=task.components.data_name,
            inputs=input_path_abs,
            input_dataset_dir=None,
            output_dataset_dir=str(npz_dir),
            require_suffix=['obj', 'fbx', 'FBX', 'dae', 'glb', 'gltf', 'vrm', 'usd', 'usda', 'usdc'],
            force_override=True,
            warning=False,
        )

        # Fix output directories to use just the filename stem, not the full nested path
        # This ensures output_dir is npz_dir/filename_stem instead of npz_dir/full/path/filename_stem
        files = [(input_file, os.path.join(str(npz_dir), input_stem)) for input_file, output_dir in files]

        # Extract mesh files to create raw_data.npz if they don't exist
        from src.data.extract import extract_builtin
        import time
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")

        # Check which files need extraction
        data_name_actual = task.components.get('data_name', 'raw_data.npz')
        files_to_extract = []
        for input_file, output_dir in files:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            raw_data_npz = os.path.join(output_dir, data_name_actual)
            if not os.path.exists(raw_data_npz):
                files_to_extract.append((input_file, output_dir))

        if files_to_extract:
            print(f"\\n=== Extracting {len(files_to_extract)} mesh file(s) ===")
            for input_file, output_dir in files_to_extract:
                print(f"  Input: {input_file}")
                print(f"  Output dir: {output_dir}")

            # Get target_count from data config if available, default to 50000
            target_count = data_config.get('faces_target_count', 50000)
            try:
                extract_builtin(
                    output_folder=str(npz_dir),
                    target_count=target_count,
                    num_runs=1,
                    id=0,
                    time=timestamp,
                    files=files_to_extract,
                )
                print("✓ Mesh extraction complete")

                # Verify extraction succeeded
                all_extracted = True
                for input_file, output_dir in files_to_extract:
                    raw_data_npz = os.path.join(output_dir, data_name_actual)
                    if os.path.exists(raw_data_npz):
                        print(f"  ✓ Verified: {raw_data_npz}")
                    else:
                        print(f"  ✗ Missing: {raw_data_npz}")
                        all_extracted = False

                if not all_extracted:
                    raise FileNotFoundError(f"Extraction failed - raw_data.npz files not found")
            except Exception as e:
                print(f"✗ Error during extraction: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            print("\\n=== No extraction needed (raw_data.npz files already exist) ===")
            # Verify files exist
            for input_file, output_dir in files:
                raw_data_npz = os.path.join(output_dir, data_name_actual)
                if not os.path.exists(raw_data_npz):
                    print(f"✗ Error: Expected raw_data.npz not found: {raw_data_npz}")
                    raise FileNotFoundError(f"raw_data.npz not found: {raw_data_npz}")
                else:
                    print(f"  ✓ Found: {raw_data_npz}")

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

        # Get model (suppress verbose warnings during loading)
        model_config = task.components.get('model', None)
        if model_config is not None:
            model_config = load_config('model', os.path.join('configs/model', model_config))
            if tokenizer_config is not None:
                tokenizer = get_tokenizer(config=tokenizer_config)
            else:
                tokenizer = None
            # Model loading will use filtered stdout/stderr automatically
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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

    finally:
        # Restore original export method
        Exporter._export_fbx = original_export_fbx

# Get configuration from JSON file
with open("config.json", 'r') as f:
    config = json.load(f)

mesh_path = config.get('mesh_path')
output_format = config.get('output_format', 'usdc')
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

    skeleton_path = export_dir / "skeleton.usdc"
    if not skeleton_path.exists():
        print(f"✗ Error: Skeleton file not found: {skeleton_path}")
        print("  Run without --skin-only to generate skeleton first")
        raise FileNotFoundError(f"Skeleton file not found: {skeleton_path}")

    # Generate skinning weights using direct function call
    print(f"Generating skinning weights for: {mesh_path}")
    print(f"Using skeleton: {skeleton_path}")

    skin_output = export_dir / "skin.usdc"
    skin_task_path = skin_task or "configs/task/quick_inference_unirig_skin.yaml"

    try:
        run_unirig_inference(
            task_path=skin_task_path,
            seed=seed,
            input_path=str(skeleton_path),
            output_path=str(skin_output),  # Export directly as USD
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

    skeleton_output = export_dir / "skeleton.usdc"
    skeleton_task_path = skeleton_task or "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"

    try:
        run_unirig_inference(
            task_path=skeleton_task_path,
            seed=seed,
            input_path=str(Path(mesh_path).resolve()),
            output_path=str(skeleton_output),  # Export directly as USD
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

    skin_output = export_dir / "skin.usdc"
    skin_task_path = skin_task or "configs/task/quick_inference_unirig_skin.yaml"

    try:
        # Use the original mesh path, not skeleton_output, to maintain directory structure
        run_unirig_inference(
            task_path=skin_task_path,
            seed=seed,
            input_path=str(Path(mesh_path).resolve()),  # Use original mesh path, not skeleton file
            output_path=str(skin_output),  # Export directly as USD
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
        # Custom merge with improved coordinate space handling
        from src.inference.merge import clean_bpy, load, process_mesh, get_arranged_bones, process_armature, merge as merge_func, get_skin
        import bpy
        import numpy as np

        # Step 1: Load skeleton from USD and extract data
        clean_bpy()
        armature = load(filepath=str(skin_output), return_armature=True)
        if armature is None:
            raise ValueError("Failed to load skeleton from USD")

        vertices_skin, faces_skin, skin = process_mesh()
        arranged_bones = get_arranged_bones(armature)
        if skin is None:
            skin = get_skin(arranged_bones)

        joints, tails, parents, names, matrix_local = process_armature(armature, arranged_bones)

        # Step 2: Load target mesh (USD) and merge
        clean_bpy()
        load(str(Path(mesh_path).resolve()))

        # Remove any existing armatures
        for c in bpy.data.armatures:
            bpy.data.armatures.remove(c)

        # Step 3: Merge skeleton and mesh
        # The merge function uses get_correct_orientation_kdtree to align skeleton with mesh
        merge_func(
            path=str(Path(mesh_path).resolve()),
            output_path=str(final_output),
            vertices=vertices_skin,
            joints=joints,
            skin=skin,
            parents=parents,
            names=names,
            tails=tails,
            add_root=False,
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
    print(f"  - {export_dir.name}/skeleton.usdc (Skeleton)")
    print(f"  - {export_dir.name}/skin.usdc (Skinning weights)")
    print(f"  - {export_dir.name}/rigged.{output_format} (Final rigged model)")
    print(f"  - {export_dir.name}/intermediate/ (Intermediate NPZ files)")
""", %{})

IO.puts("\n=== Complete ===")
IO.puts("3D model rigging completed successfully!")
