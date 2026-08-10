$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Write-Error "未检测到 Docker。请先安装并启动 Docker Desktop。"
}

if (-not (Test-Path -LiteralPath "data")) {
  New-Item -ItemType Directory -Path "data" | Out-Null
}

if (-not (Test-Path -LiteralPath "config.json")) {
  @"
{
  "auth-key": "12345678"
}
"@ | Set-Content -LiteralPath "config.json" -Encoding UTF8
  Write-Host "已创建默认 config.json，默认 auth-key 为 12345678。"
}

docker compose up -d --build

Write-Host ""
Write-Host "已启动 chatgpt2api。"
Write-Host "Web: http://localhost:3000"
Write-Host "API: http://localhost:3000/v1"
Write-Host "默认 API Key: 12345678"
