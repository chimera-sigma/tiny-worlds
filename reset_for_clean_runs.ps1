# Reset for Clean Experimental Runs
# Archives old messy logs and prepares for canonical 20-seed experiments

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Resetting for Clean Experimental Runs" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Create archive directory with timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$archiveDir = ".\experiments\results\archive_$timestamp"

Write-Host "`nCreating archive directory: $archiveDir" -ForegroundColor Yellow
New-Item -ItemType Directory -Path $archiveDir -Force | Out-Null

# Archive old summary.jsonl if it exists
if (Test-Path ".\experiments\results\summary.jsonl") {
    Write-Host "Archiving old summary.jsonl..." -ForegroundColor Yellow
    Move-Item ".\experiments\results\summary.jsonl" "$archiveDir\summary.jsonl"
    Write-Host "  -> Moved to $archiveDir\summary.jsonl" -ForegroundColor Green
} else {
    Write-Host "No summary.jsonl to archive (already clean)" -ForegroundColor Gray
}

# Archive old aggregated_results.csv if it exists
if (Test-Path ".\experiments\results\aggregated_results.csv") {
    Write-Host "Archiving old aggregated_results.csv..." -ForegroundColor Yellow
    Move-Item ".\experiments\results\aggregated_results.csv" "$archiveDir\aggregated_results.csv"
    Write-Host "  -> Moved to $archiveDir\aggregated_results.csv" -ForegroundColor Green
} else {
    Write-Host "No aggregated_results.csv to archive (already clean)" -ForegroundColor Gray
}

# Archive old runs directory if it exists and has content
if ((Test-Path ".\experiments\results\runs") -and (Get-ChildItem ".\experiments\results\runs" -File).Count -gt 0) {
    Write-Host "Archiving old per-run JSON files..." -ForegroundColor Yellow
    Move-Item ".\experiments\results\runs" "$archiveDir\runs"
    Write-Host "  -> Moved to $archiveDir\runs" -ForegroundColor Green
    # Recreate empty runs directory
    New-Item -ItemType Directory -Path ".\experiments\results\runs" -Force | Out-Null
} else {
    Write-Host "No old runs to archive" -ForegroundColor Gray
}

# Archive old figures directory if it exists
if ((Test-Path ".\experiments\results\figures") -and (Get-ChildItem ".\experiments\results\figures" -File).Count -gt 0) {
    Write-Host "Archiving old figures..." -ForegroundColor Yellow
    Move-Item ".\experiments\results\figures" "$archiveDir\figures"
    Write-Host "  -> Moved to $archiveDir\figures" -ForegroundColor Green
    # Recreate empty figures directory
    New-Item -ItemType Directory -Path ".\experiments\results\figures" -Force | Out-Null
} else {
    Write-Host "No old figures to archive" -ForegroundColor Gray
}

Write-Host "`n============================================" -ForegroundColor Green
Write-Host "Reset Complete! Ready for Clean Runs" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green

Write-Host "`nOld data archived to:" -ForegroundColor Cyan
Write-Host "  $archiveDir" -ForegroundColor White

Write-Host "`nFresh state:" -ForegroundColor Cyan
Write-Host "  - summary.jsonl: " -NoNewline -ForegroundColor White
if (Test-Path ".\experiments\results\summary.jsonl") {
    Write-Host "EXISTS (unexpected!)" -ForegroundColor Red
} else {
    Write-Host "CLEAN" -ForegroundColor Green
}
Write-Host "  - aggregated_results.csv: " -NoNewline -ForegroundColor White
if (Test-Path ".\experiments\results\aggregated_results.csv") {
    Write-Host "EXISTS (unexpected!)" -ForegroundColor Red
} else {
    Write-Host "CLEAN" -ForegroundColor Green
}
Write-Host "  - runs directory: " -NoNewline -ForegroundColor White
if ((Test-Path ".\experiments\results\runs") -and (Get-ChildItem ".\experiments\results\runs" -File).Count -eq 0) {
    Write-Host "EMPTY" -ForegroundColor Green
} else {
    Write-Host "READY" -ForegroundColor Green
}

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  1. Run clean experiments:" -ForegroundColor White
Write-Host "     .\run_clean_experiments.ps1" -ForegroundColor Yellow
Write-Host "  2. Aggregate results:" -ForegroundColor White
Write-Host "     python scripts\aggregate_results.py" -ForegroundColor Yellow
Write-Host "  3. Generate figures:" -ForegroundColor White
Write-Host "     python scripts\analyze_results.py" -ForegroundColor Yellow
Write-Host "  4. Compile paper:" -ForegroundColor White
Write-Host "     cd paper; .\compile.sh" -ForegroundColor Yellow

Write-Host "`n============================================" -ForegroundColor Cyan
