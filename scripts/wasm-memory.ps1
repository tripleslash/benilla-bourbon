#Requires -Version 5.1
#
# wasm-memory.ps1 - say what a wasm module's linear memory is, and which target features it was built with.
# The PowerShell twin of scripts/wasm-memory.py; that header holds the why, and the twins change together.
#
#   powershell -ExecutionPolicy Bypass -File scripts\wasm-memory.ps1 web\dist\wenilla_bg.wasm

param([Parameter(Position = 0)][string]$Path)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrEmpty($Path)) {
    [Console]::Error.WriteLine('usage: wasm-memory.ps1 <module.wasm>')
    exit 1
}
$full = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)

function Read-U8 {
    $b = $script:fs.ReadByte()
    if ($b -lt 0) { throw 'unexpected end of module' }
    return $b
}

function Read-Uleb {
    [long]$r = 0
    $s = 0
    while ($true) {
        $b = Read-U8
        $r = $r -bor (([long]($b -band 0x7f)) -shl $s)
        if (-not ($b -band 0x80)) { return $r }
        $s += 7
    }
}

function Read-Name {
    $n = [int](Read-Uleb)
    $buf = New-Object byte[] $n
    $got = 0
    while ($got -lt $n) {
        $k = $script:fs.Read($buf, $got, $n - $got)
        if ($k -le 0) { break }
        $got += $k
    }
    return [Text.Encoding]::UTF8.GetString($buf, 0, $got)
}

function Read-Limits {
    $flags = Read-U8
    $mn = Read-Uleb
    $mx = $null
    if ($flags -band 0x01) { $mx = Read-Uleb }
    return , @($flags, $mn, $mx)
}

function Format-Memory($limits) {
    $flags = $limits[0]
    $mn = $limits[1]
    $mx = $limits[2]
    $shared = 'no'
    if ($flags -band 0x02) { $shared = 'YES' }
    $max = 'None'
    if ($null -ne $mx) { $max = '{0} pages ({1} MiB)' -f $mx, [long][Math]::Floor($mx * 64 / 1024) }
    return ('flags=0x{0:x2} shared={1} min={2} pages ({3} MiB) max={4}' -f $flags, $shared, $mn, [long][Math]::Floor($mn * 64 / 1024), $max)
}

$script:fs = New-Object IO.FileStream($full, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read, 65536)
$found = $false
try {
    $magic = New-Object byte[] 4
    $got = $script:fs.Read($magic, 0, 4)
    if ($got -ne 4 -or $magic[0] -ne 0 -or $magic[1] -ne 0x61 -or $magic[2] -ne 0x73 -or $magic[3] -ne 0x6d) {
        [Console]::Error.WriteLine('not a wasm module')
        exit 1
    }
    [void]$script:fs.Seek(4, [IO.SeekOrigin]::Current)
    while ($true) {
        $sid = $script:fs.ReadByte()
        if ($sid -lt 0) { break }
        $size = Read-Uleb
        $start = $script:fs.Position
        if ($sid -eq 2) {
            # imports
            $count = Read-Uleb
            for ($i = 0; $i -lt $count; $i++) {
                $m = Read-Name
                $n = Read-Name
                $kind = Read-U8
                if ($kind -eq 0x00) {
                    [void](Read-Uleb)
                } elseif ($kind -eq 0x01) {
                    [void]$script:fs.ReadByte()
                    [void](Read-Limits)
                } elseif ($kind -eq 0x02) {
                    $limits = Read-Limits
                    $found = $true
                    Write-Output ('IMPORTED memory  {0}.{1}  {2}' -f $m, $n, (Format-Memory $limits))
                } elseif ($kind -eq 0x03) {
                    [void]$script:fs.ReadByte()
                    [void]$script:fs.ReadByte()
                } elseif ($kind -eq 0x04) {
                    # A tag: unread, the walk would lose its place.
                    [void]$script:fs.ReadByte()
                    [void](Read-Uleb)
                }
            }
        } elseif ($sid -eq 5) {
            # memory
            $count = Read-Uleb
            for ($i = 0; $i -lt $count; $i++) {
                $limits = Read-Limits
                $found = $true
                Write-Output ('DEFINED memory   ' + (Format-Memory $limits))
            }
        } elseif ($sid -eq 0) {
            # custom
            $here = $script:fs.Position
            $cn = Read-Name
            if ($cn -ceq 'target_features') {
                $cnt = Read-Uleb
                $feats = New-Object 'System.Collections.Generic.List[string]'
                for ($i = 0; $i -lt $cnt; $i++) {
                    $pfx = Read-U8
                    $sign = '-'
                    if ($pfx -eq 0x2b) { $sign = '+' }
                    $feats.Add($sign + (Read-Name))
                }
                $sorted = $feats.ToArray()
                [Array]::Sort($sorted, [StringComparer]::Ordinal)
                Write-Output ('target_features: ' + ($sorted -join ' '))
            }
            [void]$script:fs.Seek($here, [IO.SeekOrigin]::Begin)
        }
        [void]$script:fs.Seek($start + $size, [IO.SeekOrigin]::Begin)
    }
} finally {
    $script:fs.Dispose()
}
if (-not $found) {
    Write-Output 'no memory declaration found'
}
exit 0
