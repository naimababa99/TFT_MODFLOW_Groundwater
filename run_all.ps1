<#
.SYNOPSIS
    Runs the full leak-fixed pipeline in order and leaves it running unattended:
      1. baseline_forecasts.py   (fast, independent -- persistence/seasonal/AR benchmarks)
      2. _run_modflow_vf.py      (MODFLOW calibration, train-only objective -- produces
                                   models/modflow_enhanced/enhanced_predictions.csv)
      3. _run_hybrid_tft.py      (Hybrid TFT-MODFLOW training -- the long step)

    Each step's full console output is logged to logs/<step>_<timestamp>.log so you
    can walk away and check progress later without keeping a terminal window open
    to watch it.

.USAGE
    Open PowerShell in this folder (or anywhere) and run:
        powershell -ExecutionPolicy Bypass -File run_all.ps1

    To run just one step, e.g. only the benchmarks:
        powershell -ExecutionPolicy Bypass -File run_all.ps1 -Steps baseline

    Valid -Steps values (comma-separated): baseline, modflow, hybrid, all (default: all)
#>

param(
    [string]$Steps = "all"
)

$ErrorActionPreference = "Stop"

# Force Python into UTF-8 mode. Without this, stdout/stderr fall back to the
# system ANSI codepage (cp1252) once redirected through Tee-Object, and any
# script printing box-drawing characters (e.g. the "#" banners) or unicode
# symbols like check marks crashes with UnicodeEncodeError.
$env:PYTHONUTF8 = "1"

# Always run relative to this script's own folder, regardless of where it's invoked from.
Set-Location -Path $PSScriptRoot

$logDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

function Run-Step {
    param(
        [string]$Name,
        [string]$ScriptFile
    )

    $logFile = Join-Path $logDir "$($Name)_$timestamp.log"
    Write-Host ""
    Write-Host "==============================================================" -ForegroundColor Cyan
    Write-Host "  STARTING: $Name" -ForegroundColor Cyan
    Write-Host "  Script:   $ScriptFile" -ForegroundColor Cyan
    Write-Host "  Log:      $logFile" -ForegroundColor Cyan
    Write-Host "  Started:  $(Get-Date)" -ForegroundColor Cyan
    Write-Host "==============================================================" -ForegroundColor Cyan

    $stepStart = Get-Date

    # In Windows PowerShell 5.1, redirecting a native command's stderr with
    # *>&1/2>&1 wraps each stderr line in an ErrorRecord (NativeCommandError),
    # even for harmless warnings the script itself doesn't treat as fatal.
    # With $ErrorActionPreference = "Stop" (set above) that turns any single
    # warning line (e.g. a library version-mismatch notice) into a script-
    # terminating exception, aborting the whole pipeline even though the
    # Python process would have exited 0. Scope it to "Continue" just for
    # this invocation so only $exitCode (checked below) decides pass/fail.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        # -u = unbuffered stdout, so the log file fills in real time instead
        # of only appearing when the whole step finishes.
        & python -u $ScriptFile *>&1 | Tee-Object -FilePath $logFile
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEAP
    }

    $elapsed = (Get-Date) - $stepStart

    if ($exitCode -ne 0) {
        Write-Host ""
        Write-Host "  FAILED: $Name (exit code $exitCode, elapsed $elapsed)" -ForegroundColor Red
        Write-Host "  See log: $logFile" -ForegroundColor Red
        return $false
    }

    Write-Host ""
    Write-Host "  DONE: $Name (elapsed $elapsed)" -ForegroundColor Green
    return $true
}

$requestedSteps = $Steps.Split(",") | ForEach-Object { $_.Trim().ToLower() }
$runAll = $requestedSteps -contains "all"

$overallStart = Get-Date
$results = @{}

# --- Step 1: baseline benchmarks (fast, independent -- run first for quick feedback) ---
if ($runAll -or $requestedSteps -contains "baseline") {
    $results["baseline"] = Run-Step -Name "baseline" -ScriptFile "baseline_forecasts.py"
}

# --- Step 2: MODFLOW calibration (train-only objective) ---
if ($runAll -or $requestedSteps -contains "modflow") {
    $results["modflow"] = Run-Step -Name "modflow" -ScriptFile "_run_modflow_vf.py"
}

# --- Step 3: Hybrid TFT-MODFLOW training (the long one) ---
if ($runAll -or $requestedSteps -contains "hybrid") {
    if ($runAll -and $results.ContainsKey("modflow") -and -not $results["modflow"]) {
        Write-Host ""
        Write-Host "  SKIPPING hybrid: modflow step failed, enhanced_predictions.csv" -ForegroundColor Yellow
        Write-Host "  may be missing/stale. Fix the modflow step, then rerun with:" -ForegroundColor Yellow
        Write-Host "    powershell -ExecutionPolicy Bypass -File run_all.ps1 -Steps hybrid" -ForegroundColor Yellow
        $results["hybrid"] = $false
    } else {
        $results["hybrid"] = Run-Step -Name "hybrid" -ScriptFile "_run_hybrid_tft.py"
    }
}

$overallElapsed = (Get-Date) - $overallStart

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  PIPELINE SUMMARY (total elapsed: $overallElapsed)" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
foreach ($key in $results.Keys) {
    $status = if ($results[$key]) { "OK" } else { "FAILED" }
    $color = if ($results[$key]) { "Green" } else { "Red" }
    Write-Host ("  {0,-10} {1}" -f $key, $status) -ForegroundColor $color
}
Write-Host "  Logs in: $logDir" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
