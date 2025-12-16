param(
    [string]$SummaryPath = "experiments\results\summary.jsonl",
    [string]$OutputPath  = "experiments\results\aggregated_results.csv"
)

Write-Host "[AGG] Reading summary log from: $SummaryPath"

if (-not (Test-Path $SummaryPath)) {
    Write-Error "Summary file not found at: $SummaryPath. Did you patch logging and run experiments?"
    exit 1
}

# Read JSONL and convert each line to a PSObject
$rows = Get-Content $SummaryPath | Where-Object { $_.Trim().Length -gt 0 } | ForEach-Object {
    try {
        $_ | ConvertFrom-Json
    } catch {
        Write-Warning "Failed to parse line as JSON: $_"
        $null
    }
} | Where-Object { $_ -ne $null }

if (-not $rows) {
    Write-Error "No valid rows parsed from summary. Aborting."
    exit 1
}

Write-Host "[AGG] Parsed $($rows.Count) run entries."

# Group by world/model/hidden_dim and readout_only flag
$grouped = $rows | Group-Object -Property world_id, model_id, hidden_dim, readout_only

$aggregated = @()

foreach ($g in $grouped) {
    $key = $g.Name  # "world_id, model_id, hidden_dim, readout_only"
    $first = $g.Group[0]

    $world_id    = $first.world_id
    $world_type  = $first.world_type
    $model_id    = $first.model_id
    $model_type  = $first.model_type
    $hidden_dim  = $first.hidden_dim
    $readoutOnly = [bool]$first.readout_only

    # ΔNLL stats
    $deltaNllValues = $g.Group | ForEach-Object { $_.delta_nll_mean } | Where-Object { $_ -ne $null }
    $deltaNllStat   = $deltaNllValues | Measure-Object -Average -Minimum -Maximum

    # Probe stats (may be null/NaN in some worlds)
    $r2TrainValues = $g.Group | ForEach-Object { $_.probe_r2_trained_mean } | Where-Object { $_ -ne $null }
    $r2TrainStat   = $r2TrainValues | Measure-Object -Average -Minimum -Maximum

    $r2RandValues  = $g.Group | ForEach-Object { $_.probe_r2_random_mean } | Where-Object { $_ -ne $null }
    $r2RandStat    = $r2RandValues | Measure-Object -Average -Minimum -Maximum

    $r2PermValues  = $g.Group | ForEach-Object { $_.probe_r2_permuted_mean } | Where-Object { $_ -ne $null }
    $r2PermStat    = $r2PermValues | Measure-Object -Average -Minimum -Maximum

    # p-value for ΔNLL (just take mean across runs, if repeated)
    $pValues = $g.Group | ForEach-Object { $_.delta_nll_p_perm } | Where-Object { $_ -ne $null }
    $pMean   = ($pValues | Measure-Object -Average).Average

    $aggregated += [pscustomobject]@{
        world_id                    = $world_id
        world_type                  = $world_type
        model_id                    = $model_id
        model_type                  = $model_type
        hidden_dim                  = $hidden_dim
        readout_only                = $readoutOnly
        n_runs                      = $g.Count

        delta_nll_mean              = [double]::Parse("{0:F6}" -f $deltaNllStat.Average)
        delta_nll_min               = [double]::Parse("{0:F6}" -f $deltaNllStat.Minimum)
        delta_nll_max               = [double]::Parse("{0:F6}" -f $deltaNllStat.Maximum)
        delta_nll_p_perm_mean       = if ($pMean -ne $null) { [double]::Parse("{0:F6}" -f $pMean) } else { $null }

        probe_r2_trained_mean       = if ($r2TrainStat.Average -ne $null) { [double]::Parse("{0:F6}" -f $r2TrainStat.Average) } else { $null }
        probe_r2_trained_min        = if ($r2TrainStat.Minimum -ne $null) { [double]::Parse("{0:F6}" -f $r2TrainStat.Minimum) } else { $null }
        probe_r2_trained_max        = if ($r2TrainStat.Maximum -ne $null) { [double]::Parse("{0:F6}" -f $r2TrainStat.Maximum) } else { $null }

        probe_r2_random_mean        = if ($r2RandStat.Average -ne $null) { [double]::Parse("{0:F6}" -f $r2RandStat.Average) } else { $null }
        probe_r2_random_min         = if ($r2RandStat.Minimum -ne $null) { [double]::Parse("{0:F6}" -f $r2RandStat.Minimum) } else { $null }
        probe_r2_random_max         = if ($r2RandStat.Maximum -ne $null) { [double]::Parse("{0:F6}" -f $r2RandStat.Maximum) } else { $null }

        probe_r2_permuted_mean      = if ($r2PermStat.Average -ne $null) { [double]::Parse("{0:F6}" -f $r2PermStat.Average) } else { $null }
        probe_r2_permuted_min       = if ($r2PermStat.Minimum -ne $null) { [double]::Parse("{0:F6}" -f $r2PermStat.Minimum) } else { $null }
        probe_r2_permuted_max       = if ($r2PermStat.Maximum -ne $null) { [double]::Parse("{0:F6}" -f $r2PermStat.Maximum) } else { $null }
    }
}

# Ensure output dir exists
$outDir = Split-Path $OutputPath -Parent
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

Write-Host "[AGG] Writing aggregated results to: $OutputPath"
$aggregated | Sort-Object world_id, model_id, hidden_dim, readout_only | Export-Csv -Path $OutputPath -NoTypeInformation

Write-Host "[AGG] Done. Aggregated $($aggregated.Count) grouped entries."
