param(
    [string]$Prefix = "$env:USERPROFILE\conda-envs\cs699-saferl",
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"

Write-Host "Creating conda environment at $Prefix"
conda create -y -p $Prefix python=3.10

$python = Join-Path $Prefix "python.exe"
if (-not (Test-Path $python)) {
    throw "Could not find python.exe inside $Prefix"
}

if ($CpuOnly) {
    Write-Host "Installing CPU PyTorch"
    & $python -m pip install --no-cache-dir torch
} else {
    Write-Host "Installing CUDA PyTorch (cu118)"
    & $python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118
}

Write-Host "Installing project dependencies"
& $python -m pip install --no-cache-dir `
    "numpy<2" `
    gymnasium `
    highway-env `
    tianshou `
    matplotlib `
    "imageio[ffmpeg]" `
    scipy `
    pandas `
    seaborn `
    fast-safe-rl --no-deps `
    pyrallis==0.3.1 `
    pyyaml~=6.0 `
    wandb~=0.14.0 `
    protobuf~=3.19.0 `
    prettytable~=3.7.0 `
    typing-inspect

& $python -m pip install --no-cache-dir `
    cloudpickle `
    farama-notifications `
    pygame `
    tqdm `
    tensorboard `
    numba `
    h5py `
    pettingzoo `
    contourpy `
    cycler `
    fonttools `
    kiwisolver `
    pillow `
    pyparsing `
    python-dateutil `
    pytz `
    tzdata `
    click `
    GitPython `
    requests `
    sentry-sdk `
    docker-pycreds `
    pathtools `
    setproctitle `
    appdirs `
    wcwidth `
    mypy-extensions `
    charset_normalizer `
    idna `
    urllib3 `
    certifi `
    six `
    gitdb `
    smmap `
    absl-py `
    grpcio `
    markdown `
    tensorboard-data-server `
    werkzeug `
    imageio-ffmpeg `
    psutil `
    colorama

& $python -m pip install --no-cache-dir "setuptools<81"

Write-Host ""
Write-Host "Environment ready."
Write-Host "Activate with:"
Write-Host "  conda activate $Prefix"
