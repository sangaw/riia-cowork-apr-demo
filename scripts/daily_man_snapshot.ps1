# RITA — Daily Manoeuvre Snapshot Trigger
# Runs at 3:35 PM IST (10:05 UTC) via Windows Task Scheduler.
# Calls POST /api/v1/portfolio/man-daily-snapshot for each of the 3 active NSE expiry months.
#
# Setup (run once as Administrator):
#   schtasks /create /tn "RITA-ManSnapshot" /tr "powershell -ExecutionPolicy Bypass -File \"C:\Users\Sandeep\Documents\Work\code\poc\rita-cowork-demo\scripts\daily_man_snapshot.ps1\"" /sc DAILY /st 15:35 /f
#
# To delete:
#   schtasks /delete /tn "RITA-ManSnapshot" /f

$API_BASE = "http://localhost:8000"
$LOG_DIR  = "C:\Users\Sandeep\Documents\Work\code\poc\rita-cowork-demo\rita_output\logs"
$LOG_FILE = Join-Path $LOG_DIR ("man_snapshot_" + (Get-Date -Format "yyyy-MM") + ".log")

# Ensure log directory exists
if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR | Out-Null }

function Write-Log {
    param([string]$msg)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Add-Content -Path $LOG_FILE -Value $line
    Write-Host $line
}

# ── NSE last-Thursday expiry logic ───────────────────────────────────────────
function Get-LastThursday {
    param([int]$Year, [int]$Month)
    # Find last Thursday of given month
    $lastDay = [DateTime]::new($Year, $Month, [DateTime]::DaysInMonth($Year, $Month))
    $offset  = ($lastDay.DayOfWeek - [DayOfWeek]::Thursday + 7) % 7
    return $lastDay.AddDays(-$offset)
}

function Get-ActiveMonths {
    $today    = [DateTime]::Today
    $abbrevs  = @("","JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC")
    $months   = @()
    $year     = $today.Year
    $month    = $today.Month

    # If today is past last Thursday of current month, advance window by 1
    $expiry = Get-LastThursday $year $month
    if ($today -gt $expiry) {
        $month++
        if ($month -gt 12) { $month = 1; $year++ }
    }

    for ($i = 0; $i -lt 3; $i++) {
        $m = $month + $i
        $y = $year
        if ($m -gt 12) { $m -= 12; $y++ }
        $months += $abbrevs[$m]
    }
    return $months
}

# ── Main ─────────────────────────────────────────────────────────────────────
Write-Log "=== RITA Daily Manoeuvre Snapshot ==="

# Check API is reachable
try {
    $health = Invoke-RestMethod -Uri "$API_BASE/health" -Method Get -TimeoutSec 5
    Write-Log "API reachable — version $($health.version)"
} catch {
    Write-Log "ERROR: API not reachable at $API_BASE — is run_api.py running? Aborting."
    exit 1
}

$activeMonths = Get-ActiveMonths
Write-Log "Active months: $($activeMonths -join ', ')"

foreach ($month in $activeMonths) {
    try {
        $body    = @{ month = $month } | ConvertTo-Json
        $headers = @{ "Content-Type" = "application/json" }
        $resp    = Invoke-RestMethod -Uri "$API_BASE/api/v1/portfolio/man-daily-snapshot" `
                                     -Method Post -Body $body -Headers $headers -TimeoutSec 30
        $status  = $resp.status
        if ($status -eq "skipped") {
            Write-Log "  $month — skipped ($($resp.reason))"
        } else {
            Write-Log "  $month — OK  groups=$($resp.groups_written)  rows=$($resp.rows_written)"
        }
    } catch {
        Write-Log "  $month — FAILED: $($_.Exception.Message)"
    }
}

Write-Log "Done."
