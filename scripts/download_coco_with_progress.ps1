# Stream-download COCO 2017 with explicit progress prints (works fine in
# non-TTY shells where the built-in Invoke-WebRequest progress bar disappears).
param(
    [string]$DestDir = "vl_jepa\data\COCO2017",
    [string[]]$Only = @()  # e.g. -Only annotations,val
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# Windows PowerShell 5.1 doesn't auto-load System.Net.Http. Make sure it's there.
Add-Type -AssemblyName System.Net.Http -ErrorAction SilentlyContinue
# Some Windows builds default to TLS 1.0; cocodataset.org needs TLS 1.2+.
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12

$files = @(
    @{ name = "annotations"; url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"; out = "annotations_trainval2017.zip" }
    @{ name = "val";         url = "http://images.cocodataset.org/zips/val2017.zip";                         out = "val2017.zip" }
    @{ name = "train";       url = "http://images.cocodataset.org/zips/train2017.zip";                       out = "train2017.zip" }
)

if ($Only.Count -gt 0) {
    $files = $files | Where-Object { $Only -contains $_.name }
}

New-Item -ItemType Directory -Force -Path $DestDir | Out-Null

function Download-WithProgress {
    param([string]$Url, [string]$OutPath, [string]$Label)

    Write-Host ""
    Write-Host "=== Downloading $Label ===" -ForegroundColor Cyan
    Write-Host "From: $Url"
    Write-Host "To:   $OutPath"

    $client = New-Object System.Net.Http.HttpClient
    $client.Timeout = [System.TimeSpan]::FromHours(4)
    try {
        $response = $client.GetAsync($Url, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
        $response.EnsureSuccessStatusCode() | Out-Null

        $total = $response.Content.Headers.ContentLength
        $totalMB = if ($total) { [math]::Round($total / 1MB, 1) } else { 0 }
        Write-Host ("Size: {0} MB" -f $totalMB)

        $remote = $response.Content.ReadAsStreamAsync().Result
        $local  = [System.IO.File]::Open($OutPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
        $buffer = New-Object byte[] (1MB)
        $read = 0L
        $start = Get-Date
        $lastPrintPct = -1
        try {
            while (($n = $remote.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $local.Write($buffer, 0, $n)
                $read += $n
                if ($total) {
                    $pct = [int](100 * $read / $total)
                    if ($pct -ne $lastPrintPct -and $pct % 2 -eq 0) {
                        $elapsed = ((Get-Date) - $start).TotalSeconds
                        $mbps = if ($elapsed -gt 0) { ($read / 1MB) / $elapsed } else { 0 }
                        $eta  = if ($mbps -gt 0) { ($total - $read) / 1MB / $mbps } else { 0 }
                        $bar = "#" * [int]($pct / 2.5) + "." * (40 - [int]($pct / 2.5))
                        Write-Host ("[{0}] {1,3}%  {2,7:N1}/{3,7:N1} MB  {4,5:N1} MB/s  ETA {5,5:N0}s" -f $bar, $pct, ($read/1MB), ($total/1MB), $mbps, $eta)
                        $lastPrintPct = $pct
                    }
                }
            }
        } finally {
            $local.Close()
            $remote.Close()
        }
        $elapsed = ((Get-Date) - $start).TotalSeconds
        Write-Host ("Done in {0:N1}s ({1:N1} MB at {2:N1} MB/s avg)" -f $elapsed, ($read/1MB), (($read/1MB)/[math]::Max($elapsed,1))) -ForegroundColor Green
    } finally {
        $client.Dispose()
    }
}

function Extract-Zip {
    param([string]$ZipPath, [string]$Dest)
    Write-Host "Extracting $(Split-Path -Leaf $ZipPath)..."
    $t0 = Get-Date
    Expand-Archive -Path $ZipPath -DestinationPath $Dest -Force
    Write-Host ("  extracted in {0:N1}s" -f ((Get-Date) - $t0).TotalSeconds) -ForegroundColor Green
}

foreach ($f in $files) {
    $outPath = Join-Path $DestDir $f.out
    if (Test-Path $outPath) {
        Write-Host "Skipping $($f.name): already exists at $outPath" -ForegroundColor Yellow
    } else {
        Download-WithProgress -Url $f.url -OutPath $outPath -Label $f.name
    }
    Extract-Zip -ZipPath $outPath -Dest $DestDir
}

Write-Host ""
Write-Host "All requested files downloaded and extracted to $DestDir" -ForegroundColor Green
