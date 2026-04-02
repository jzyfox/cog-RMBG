param(
    [string]$ModelDir = $env:QWEN3_VL_MODEL_DIR,
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 8008
)

if (-not $ModelDir) {
    $ModelDir = "J:\Models\Qwen\Qwen3-VL-8B-Instruct"
}

$env:QWEN3_VL_MODEL_DIR = $ModelDir

Write-Host "QWEN3_VL_MODEL_DIR=$($env:QWEN3_VL_MODEL_DIR)"
Write-Host "Starting local Qwen3-VL server on http://$ListenHost`:$Port"

python -m uvicorn local_qwen3_vl_server:app --host $ListenHost --port $Port
