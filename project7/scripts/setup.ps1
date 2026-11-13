$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force vendor, models, assets, output | Out-Null

function Download($Url, $Path) {
    if (!(Test-Path -LiteralPath $Path)) {
        if (Get-Command aria2c.exe -ErrorAction SilentlyContinue) {
            & aria2c.exe -x 16 -s 16 -k 1M --file-allocation=none --auto-file-renaming=false `
                -d (Split-Path $Path -Parent) -o (Split-Path $Path -Leaf) $Url
        } else {
            & curl.exe -L --fail --retry 3 -o $Path $Url
        }
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $Path" }
    }
}

if (!(Test-Path vendor/blender/blender-4.5.13-windows-x64/blender.exe)) {
    Download 'https://download.blender.org/release/Blender4.5/blender-4.5.13-windows-x64.zip' 'vendor/blender.zip'
    Expand-Archive vendor/blender.zip vendor/blender -Force
}
if (!(Test-Path vendor/sdcpp-ext/sd-cli.exe)) {
    Download 'https://github.com/leejet/stable-diffusion.cpp/releases/download/master-1086-465adf3/sd-master-465adf3-bin-win-cpu-x64.zip' 'vendor/sdcpp.zip'
    Expand-Archive vendor/sdcpp.zip vendor/sdcpp-ext -Force
}

Download 'https://hf-mirror.com/samoudi/sd-turbo-gguf/resolve/main/sd_turbo_q4_0.gguf?download=true' 'models/sd_turbo-q4_0.gguf'
if ((Get-FileHash 'models/sd_turbo-q4_0.gguf' -Algorithm SHA256).Hash -ne 'D069EF2B2D3D5C2465662D706C242A1016DB56163C109BD7E3008A94CDD00359') {
    throw 'Checksum failed: models/sd_turbo-q4_0.gguf'
}
if (!(Test-Path models/control_depth_sd21-fp16.gguf)) {
    Download 'https://hf-mirror.com/thibaud/controlnet-sd21/resolve/main/control_v11p_sd21_depth.safetensors?download=true' 'models/control_v11p_sd21_depth.safetensors'
    if ((Get-FileHash 'models/control_v11p_sd21_depth.safetensors' -Algorithm SHA256).Hash -ne 'ED33EC158B6EA3BF61494EE71ADC16D45886D5F6611D108345A184EBE7D08D81') {
        throw 'Checksum failed: models/control_v11p_sd21_depth.safetensors'
    }
    & ./vendor/sdcpp-ext/sd-cli.exe -M convert -m models/control_v11p_sd21_depth.safetensors `
        --type f16 -o models/control_depth_sd21-fp16.gguf
    if ($LASTEXITCODE -ne 0) { throw 'ControlNet conversion failed' }
}
