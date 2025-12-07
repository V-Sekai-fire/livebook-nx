# merge texture
require_suffix="obj,fbx,FBX,dae,glb,gltf,vrm"
source=""
target=""
output=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --require_suffix) require_suffix="$2"; shift ;;
        --source) source="$2"; shift ;;
        --target) target="$2"; shift ;;
        --output) output="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

cmd=" \
    python -m src.inference.merge \
    --require_suffix=$require_suffix \
    --num_runs=1 \
    --id=0 \
    --source=$source \
    --target=$target \
    --output=$output \
"

cmd="$cmd &"
eval $cmd

wait

echo "done"