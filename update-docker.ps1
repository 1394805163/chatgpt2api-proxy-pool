$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

git pull
docker compose up -d --build

Write-Host ""
Write-Host "已拉取最新代码并重建容器。"
