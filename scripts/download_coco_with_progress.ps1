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
    param([string]$Url, [string]$OutPath, [string]$Label, [int]$MaxRetries = 6)

    Write-Host ""
    Write-Host "=== Downloading $Label ===" -ForegroundColor Cyan
    Write-Host "From: $Url"
    Write-Host "To:   $OutPath"

    $attempt = 0
    while ($true) {
        $attempt++
        $offset = 0L
        if (Test-Path $OutPath) {
            $offset = (Get-Item $OutPath).Length
            if ($offset -gt 0) {
                Write-Host ("Resuming from {0:N1} MB already on disk" -f ($offset/1MB)) -ForegroundColor Yellow
            }
        }

        $client = New-Object System.Net.Http.HttpClient
        $client.Timeout = [System.TimeSpan]::FromHours(4)
        try {
            $req = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Get, $Url)
            if ($offset -gt 0) {
                $req.Headers.Range = New-Object System.Net.Http.Headers.RangeHeaderValue($offset, $null)
            }

            $response = $client.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
            # 416 = Requested Range Not Satisfiable. Usually means we already
            # have the whole file, so the Range start sits past EOF.
            if ([int]$response.StatusCode -eq 416) {
                Write-Host ("Server says range not satisfiable; assuming file is already complete ({0:N1} MB on disk)" -f ($offset/1MB)) -ForegroundColor Green
                return
            }
            $response.EnsureSuccessStatusCode() | Out-Null

            # Server total: ContentRange.Length when resuming, else ContentLength + offset (=ContentLength when offset=0).
            $total = $null
            if ($response.Content.Headers.ContentRange -and $response.Content.Headers.ContentRange.Length) {
                $total = [int64]$response.Content.Headers.ContentRange.Length
            } elseif ($response.Content.Headers.ContentLength) {
                $total = [int64]$response.Content.Headers.ContentLength + $offset
            }
            $statusCode = [int]$response.StatusCode

            # If we asked for a range but server returned 200 (full body), it ignored the Range header.
            # Bail out of resume and rewrite the file from scratch.
            if ($offset -gt 0 -and $statusCode -ne 206) {
                Write-Host "Server ignored Range header (status $statusCode). Restarting from byte 0." -ForegroundColor Yellow
                $offset = 0
            }

            $totalMB = if ($total) { [math]::Round($total / 1MB, 1) } else { 0 }
            Write-Host ("Size: {0} MB (attempt {1}/{2})" -f $totalMB, $attempt, $MaxRetries)

            $remote = $response.Content.ReadAsStreamAsync().Result
            if ($offset -gt 0) {
                $local = [System.IO.File]::Open($OutPath, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write)
            } else {
                $local = [System.IO.File]::Open($OutPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
            }
            $buffer = New-Object byte[] (1MB)
            $readThisSession = 0L
            $start = Get-Date
            $lastPrintPct = -1
            $failed = $false
            try {
                while (($n = $remote.Read($buffer, 0, $buffer.Length)) -gt 0) {
                    $local.Write($buffer, 0, $n)
                    $readThisSession += $n
                    $totalRead = $offset + $readThisSession
                    if ($total) {
                        $pct = [int](100 * $totalRead / $total)
                        if ($pct -ne $lastPrintPct -and $pct % 2 -eq 0) {
                            $elapsed = ((Get-Date) - $start).TotalSeconds
                            $mbps = if ($elapsed -gt 0) { ($readThisSession / 1MB) / $elapsed } else { 0 }
                            $eta  = if ($mbps -gt 0) { ($total - $totalRead) / 1MB / $mbps } else { 0 }
                            $bar = "#" * [int]($pct / 2.5) + "." * (40 - [int]($pct / 2.5))
                            Write-Host ("[{0}] {1,3}%  {2,7:N1}/{3,7:N1} MB  {4,5:N1} MB/s  ETA {5,5:N0}s" -f $bar, $pct, ($totalRead/1MB), ($total/1MB), $mbps, $eta)
                            $lastPrintPct = $pct
                        }
                    }
                }
            } catch {
                $failed = $true
                Write-Host ""
                Write-Host ("Stream error: {0}" -f $_.Exception.Message) -ForegroundColor Red
            } finally {
                $local.Close()
                $remote.Close()
            }

            $elapsed = ((Get-Date) - $start).TotalSeconds
            $onDisk = (Get-Item $OutPath).Length

            if (-not $failed -and (-not $total -or $onDisk -ge $total)) {
                Write-Host ("Done. {0:N1} MB on disk ({1:N1} MB this session in {2:N1}s)" -f ($onDisk/1MB), ($readThisSession/1MB), $elapsed) -ForegroundColor Green
                return
            }

            if ($attempt -ge $MaxRetries) {
                throw "Giving up after $MaxRetries attempts. $($onDisk/1MB) MB on disk of expected $($total/1MB) MB."
            }

            $backoff = [math]::Min(60, 5 * [math]::Pow(2, $attempt - 1))
            Write-Host ("Will retry in {0:N0}s (resume from {1:N1} MB)..." -f $backoff, ($onDisk/1MB)) -ForegroundColor Yellow
            Start-Sleep -Seconds $backoff
        } finally {
            $client.Dispose()
        }
    }
}

function Extract-Zip {
    param([string]$ZipPath, [string]$Dest)
    # Windows PowerShell 5.1's Expand-Archive uses an older .NET that does
    # not support ZIP64, so any archive over ~4 GB blows up with
    # "Split or spanned archives are not supported." COCO train2017.zip is
    # ~18 GB, so we shell out to Python instead - zipfile handles ZIP64
    # transparently.
    $name = Split-Path -Leaf $ZipPath
    Write-Host "Extracting $name (via Python zipfile, ZIP64-safe)..."
    $t0 = Get-Date

    # Pick the first available python on PATH.
    $pyExe = $null
    foreach ($candidate in @("python", "py")) {
        $resolved = (Get-Command $candidate -ErrorAction SilentlyContinue)
        if ($resolved) { $pyExe = $resolved.Source; break }
    }
    if (-not $pyExe) {
        throw "Python not found on PATH. Install Python 3 or add it to PATH so we can extract ZIP64 archives."
    }

    # Hand the work to Python. Forward-slash the paths to dodge backslash
    # escaping inside the inline script.
    $zipFwd = $ZipPath -replace '\\', '/'
    $destFwd = $Dest   -replace '\\', '/'
    $pyCode = @"
import sys, zipfile, time
z = sys.argv[1]
d = sys.argv[2]
t0 = time.time()
with zipfile.ZipFile(z) as zf:
    names = zf.namelist()
    total = len(names)
    last = -1
    for i, n in enumerate(names, 1):
        zf.extract(n, d)
        pct = int(100 * i / total)
        if pct != last and pct % 5 == 0:
            print(f'  extract: {pct:3d}%  {i}/{total} files  {time.time()-t0:6.1f}s', flush=True)
            last = pct
print(f'  extract done: {total} files in {time.time()-t0:.1f}s', flush=True)
"@
    $tmp = [System.IO.Path]::GetTempFileName() + ".py"
    Set-Content -Path $tmp -Value $pyCode -Encoding ASCII
    try {
        & $pyExe $tmp $zipFwd $destFwd
        if ($LASTEXITCODE -ne 0) {
            throw "Python extractor exited with code $LASTEXITCODE"
        }
    } finally {
        Remove-Item $tmp -ErrorAction SilentlyContinue
    }
    Write-Host ("  extracted in {0:N1}s" -f ((Get-Date) - $t0).TotalSeconds) -ForegroundColor Green
}

foreach ($f in $files) {
    $outPath = Join-Path $DestDir $f.out
    # Always call the downloader. It checks the local file size and either
    # no-ops (file already complete), resumes (partial), or starts fresh.
    # The previous "Test-Path then skip" branch swallowed resume attempts on
    # half-finished files and skipped straight to Extract-Zip.
    Download-WithProgress -Url $f.url -OutPath $outPath -Label $f.name
    Extract-Zip -ZipPath $outPath -Dest $DestDir
}

Write-Host ""
Write-Host "All requested files downloaded and extracted to $DestDir" -ForegroundColor Green
