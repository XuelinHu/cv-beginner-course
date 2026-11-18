param([switch]$Download)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $labArguments = @('run', '--no-capture-output', '-n', 'pyg', 'python', '-u', 'scripts/run_all.py')
    if ($Download) { $labArguments += '--download' }
    & conda @labArguments
    if ($LASTEXITCODE -ne 0) { throw "课程执行失败，请检查第一处报错。" }
    Write-Host '课程执行完成：site/index.html'
} finally {
    Pop-Location
}
