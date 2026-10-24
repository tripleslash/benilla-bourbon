#Requires -Version 5.1
#
# check.ps1 - the in-loop verify: fmt everywhere, clippy and test on the changed crates and their dependents.
# The PowerShell twin of scripts/check.sh; that header holds the why, and the twins change together.
# Escalates to scripts/gates.ps1.
#
#   powershell -ExecutionPolicy Bypass -File scripts\check.ps1
#
# CHECK_FULL=1 goes straight to gates.ps1.

$ErrorActionPreference = 'Stop'
$dash = [string][char]0x2014
$times = [string][char]0x00D7
$utf8 = New-Object Text.UTF8Encoding $false

function ConvertTo-NativeArg([string]$a) {
    if ($a.Length -gt 0 -and $a -notmatch '[\s"]') { return $a }
    $sb = New-Object Text.StringBuilder
    [void]$sb.Append('"')
    $bs = 0
    foreach ($c in $a.ToCharArray()) {
        if ($c -eq '\') { $bs++; continue }
        if ($c -eq '"') { [void]$sb.Append(('\' * ($bs * 2 + 1)) + '"'); $bs = 0; continue }
        [void]$sb.Append(('\' * $bs) + $c)
        $bs = 0
    }
    [void]$sb.Append(('\' * ($bs * 2)) + '"')
    return $sb.ToString()
}

# Run a program in the current location. $EnvVars adds variables ($null unsets one; '' sets it empty,
# which $env: cannot). -LogPath merges stdout and stderr into a file, -Capture keeps stdout in
# $script:NativeOut (trailing newlines cut, as `$(...)` does), -Quiet drops both. Returns the exit
# code, 127 when the program is not found.
function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$ArgList = @(),
        [hashtable]$EnvVars = @{},
        [string]$LogPath = '',
        [switch]$Capture,
        [switch]$Quiet
    )
    $script:NativeOut = ''
    $psi = New-Object Diagnostics.ProcessStartInfo
    $psi.FileName = $Exe
    $psi.Arguments = (@($ArgList | ForEach-Object { ConvertTo-NativeArg $_ }) -join ' ')
    $psi.UseShellExecute = $false
    $psi.WorkingDirectory = (Get-Location).ProviderPath
    foreach ($k in @($EnvVars.Keys)) {
        if ($null -eq $EnvVars[$k]) { $psi.EnvironmentVariables.Remove($k) }
        else { $psi.EnvironmentVariables[$k] = [string]$EnvVars[$k] }
    }
    $toLog = $LogPath -ne ''
    $psi.RedirectStandardOutput = $toLog -or $Capture -or $Quiet
    $psi.RedirectStandardError = $toLog -or $Quiet
    if ($psi.RedirectStandardOutput) { $psi.StandardOutputEncoding = $utf8 }
    if ($psi.RedirectStandardError) { $psi.StandardErrorEncoding = $utf8 }
    $log = $null
    $out = New-Object 'System.Collections.Generic.List[string]'
    $p = New-Object Diagnostics.Process
    $p.StartInfo = $psi
    try {
        if ($toLog) {
            $log = New-Object IO.StreamWriter($LogPath, $false, $utf8)
            $log.NewLine = "`n"
        }
        try { [void]$p.Start() } catch {
            if ($toLog) { $log.WriteLine("${Exe}: command not found") }
            return 127
        }
        $readers = @{}
        if ($psi.RedirectStandardOutput) { $readers['out'] = $p.StandardOutput }
        if ($psi.RedirectStandardError) { $readers['err'] = $p.StandardError }
        $tasks = @{}
        foreach ($k in @($readers.Keys)) { $tasks[$k] = $readers[$k].ReadLineAsync() }
        while ($tasks.Count -gt 0) {
            $keys = @($tasks.Keys)
            $pending = New-Object 'System.Collections.Generic.List[System.Threading.Tasks.Task]'
            foreach ($k in $keys) { $pending.Add($tasks[$k]) }
            $i = [Threading.Tasks.Task]::WaitAny($pending.ToArray())
            $k = $keys[$i]
            $line = $tasks[$k].Result
            if ($null -eq $line) { $tasks.Remove($k); continue }
            $tasks[$k] = $readers[$k].ReadLineAsync()
            if ($toLog) { $log.WriteLine($line) }
            elseif ($Capture -and $k -eq 'out') { $out.Add($line) }
        }
        $p.WaitForExit()
        $script:NativeOut = ($out -join "`n").TrimEnd([char]10)
        return $p.ExitCode
    } finally {
        if ($null -ne $log) { $log.Dispose() }
        $p.Dispose()
    }
}

function Read-Lines([string]$path) {
    if (-not [IO.File]::Exists($path)) { return , @() }
    $text = $utf8.GetString([IO.File]::ReadAllBytes($path))
    if ($text -eq '') { return , @() }
    $lines = $text.Split([char]10)
    if ($lines[$lines.Length - 1] -eq '') { $lines = $lines[0..($lines.Length - 2)] }
    return , @($lines)
}

# mktemp in ${TMPDIR:-<the temp dir>}
function New-TempFile([string]$prefix) {
    $dir = $env:TMPDIR
    if ([string]::IsNullOrEmpty($dir)) { $dir = [IO.Path]::GetTempPath() }
    while ($true) {
        $path = [IO.Path]::Combine($dir, $prefix + '.' + [IO.Path]::GetRandomFileName().Replace('.', '').Substring(0, 6))
        try {
            $fs = New-Object IO.FileStream($path, [IO.FileMode]::CreateNew)
            $fs.Dispose()
            return $path
        } catch [IO.IOException] { }
    }
}

# Hand over to the full chain, as `exec scripts/gates.sh` does.
function Invoke-Gates {
    & ([IO.Path]::Combine($root, 'scripts\gates.ps1'))
    exit $LASTEXITCODE
}

# Check the checkout you stand in when it is one of this repo, else this file's (as gates.ps1 does).
$here = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$code = Invoke-Native 'git' @('rev-parse', '--show-toplevel') -Capture
$root = $script:NativeOut
if ($code -ne 0) { $root = '' }
if (-not $root -or -not [IO.File]::Exists([IO.Path]::Combine($root, 'scripts\check.ps1'))) { $root = $here }
Set-Location -LiteralPath $root
[Environment]::CurrentDirectory = (Get-Location).ProviderPath

if ($env:CHECK_FULL -eq '1') { Invoke-Gates }

# The round's change set: the fork point against main, plus staged, unstaged and untracked files.
if ((Invoke-Native 'git' @('merge-base', 'HEAD', 'main') -Capture) -ne 0) {
    [void](Invoke-Native 'git' @('rev-parse', 'HEAD') -Capture)
}
$base = $script:NativeOut
$set = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)
foreach ($argList in @(
        , @('diff', '--name-only', $base, 'HEAD')
        , @('diff', '--name-only', 'HEAD')
        , @('ls-files', '--others', '--exclude-standard'))) {
    [void](Invoke-Native 'git' $argList -Capture)
    foreach ($line in $script:NativeOut.Split([char]10)) { if ($line -ne '') { [void]$set.Add($line) } }
}
$changed = New-Object 'string[]' $set.Count
$set.CopyTo($changed)
[Array]::Sort($changed, [StringComparer]::Ordinal)

if ($changed.Length -eq 0) {
    Write-Output "check: no changes against main and a clean tree $dash nothing to verify"
    exit 0
}

# Map files to crate dirs; anything unmappable escalates to the full chain.
$dirSet = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)
$fullReason = ''
foreach ($f in $changed) {
    if ($f.StartsWith('docs/') -or $f.EndsWith('.md')) { continue }
    if ($f -cmatch '^crates/([^/]*)/.') { [void]$dirSet.Add('crates/' + $Matches[1]); continue }
    $fullReason = $f
}

if ($fullReason) {
    Write-Output "check: '$fullReason' is outside the crate map $dash the whole workspace could be affected."
    Write-Output 'check: escalating to the full chain: scripts/gates.ps1'
    Invoke-Gates
}

$dirlist = New-Object 'string[]' $dirSet.Count
$dirSet.CopyTo($dirlist)
[Array]::Sort($dirlist, [StringComparer]::Ordinal)
if ($dirlist.Length -eq 0) {
    Write-Output "check: docs-only round ($($changed.Length) file(s)) $dash no gate can change; fmt only"
    if ((Invoke-Native 'cargo' @('fmt', '--all', '--', '--check')) -ne 0) { exit 1 }
    Write-Output 'check: green (docs only)'
    exit 0
}

# Changed crates and their reverse-dependency closure, off cargo metadata.
if ((Invoke-Native 'cargo' @('metadata', '--format-version', '1', '--no-deps') -Capture -Quiet) -ne 0) {
    Write-Output "check: cargo metadata failed $dash falling back to the full chain"
    Invoke-Gates
}
$meta = $null
try { $meta = $script:NativeOut | ConvertFrom-Json } catch { }
if ($null -eq $meta) {
    Write-Output "check: cargo metadata failed $dash falling back to the full chain"
    Invoke-Gates
}
$rootFull = [IO.Path]::GetFullPath($root).TrimEnd('\', '/')
$names = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)
foreach ($p in $meta.packages) { [void]$names.Add([string]$p.name) }
$byDir = New-Object 'System.Collections.Generic.Dictionary[string,string]' ([StringComparer]::Ordinal)
$deps = New-Object 'System.Collections.Generic.Dictionary[string,object]' ([StringComparer]::Ordinal)
foreach ($p in $meta.packages) {
    $dir = [IO.Path]::GetDirectoryName([IO.Path]::GetFullPath([string]$p.manifest_path))
    if ($dir.StartsWith($rootFull + '\', [StringComparison]::OrdinalIgnoreCase)) {
        $rel = $dir.Substring($rootFull.Length + 1)
    } else {
        $rel = $dir
    }
    $byDir[$rel.Replace('\', '/')] = [string]$p.name
    $ds = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)
    foreach ($d in @($p.dependencies)) { if ($null -ne $d -and $names.Contains([string]$d.name)) { [void]$ds.Add([string]$d.name) } }
    $deps[[string]$p.name] = $ds
}
$missing = @($dirlist | Where-Object { -not $byDir.ContainsKey($_) })
if ($missing.Count -gt 0) {
    Write-Output "check: changed path(s) not in the workspace map ($($missing -join ' ')) $dash full chain"
    Invoke-Gates
}
$scope = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)
foreach ($d in $dirlist) { [void]$scope.Add($byDir[$d]) }
$grew = $true
while ($grew) {
    $grew = $false
    foreach ($name in @($deps.Keys)) {
        if ($scope.Contains($name)) { continue }
        if ($deps[$name].Overlaps($scope)) { [void]$scope.Add($name); $grew = $true }
    }
}
$pkgList = New-Object 'string[]' $scope.Count
$scope.CopyTo($pkgList)
[Array]::Sort($pkgList, [StringComparer]::Ordinal)
$pkgs = $pkgList -join ' '

$pflags = New-Object 'System.Collections.Generic.List[string]'
foreach ($p in $pkgList) { $pflags.Add('-p'); $pflags.Add($p) }
Write-Output "check: scope = $pkgs"
Write-Output 'check:   (changed crates + everything that depends on them; the full chain runs once, at land)'

$log = New-TempFile 'benilla-check'
$skips = New-TempFile 'benilla-check-skips'
try {
    # Each gate logs to a file read afterwards.
    function Invoke-Step([string]$name, [string[]]$argList, [hashtable]$envMods = @{}) {
        if ((Invoke-Native 'cargo' $argList -EnvVars $envMods -LogPath $log) -ne 0) {
            Write-Output "CHECK FAILED: $name"
            # The failing test or error first: a test gate's tail is the doc-test noise after it.
            $lines = Read-Lines $log
            @($lines | Where-Object { $_ -cmatch 'panicked at|^---- .* ----$|^test .* FAILED$|^error(\[E[0-9]+\])?:' }) |
                Select-Object -First 40 | ForEach-Object { Write-Output $_ }
            $lines | Select-Object -Last 30 | ForEach-Object { Write-Output $_ }
            exit 1
        }
        Write-Output "check ok: $name"
    }

    Invoke-Step 'fmt' @('fmt', '--all', '--', '--check')
    Invoke-Step 'clippy' (@('clippy') + $pflags.ToArray() + @('--all-targets', '--', '-D', 'warnings'))
    # --no-fail-fast: one red test binary does not hide the failures in the ones after it.
    Invoke-Step 'test' (@('test', '--no-fail-fast') + $pflags.ToArray()) @{ BENILLA_SKIP_LOG = $skips }
    # Data-gated tests skip without the install or the addon corpus and libtest swallows their line.
    $lines = Read-Lines $skips
    if ($lines.Count -gt 0) {
        Write-Output "  test: $($lines.Count) data-gated tests SKIPPED on this machine $dash"
        $counts = New-Object 'System.Collections.Generic.Dictionary[string,int]'
        $rows = New-Object 'System.Collections.Generic.List[string]'
        foreach ($l in $lines) {
            if ($counts.ContainsKey($l)) { $counts[$l]++ } else { $counts[$l] = 1; $rows.Add($l) }
        }
        # sort | uniq -c | sort -rn: the most frequent first, ties in reverse byte order.
        $rows.Sort([Comparison[string]] {
                param($x, $y)
                if ($counts[$x] -ne $counts[$y]) { return $counts[$y] - $counts[$x] }
                return [string]::CompareOrdinal($y, $x)
            })
        foreach ($r in $rows) { Write-Output "    $($counts[$r]) $times $r" }
        Write-Output '    (they run where the data is: docs/CONTRIBUTING.md, "Setting up")'
    }

    Write-Output "CHECK GREEN (scoped: $pkgs)"
    Write-Output '  (the full chain, scripts/gates.ps1, runs once on the tree that lands; a run before it repeats it)'
    exit 0
} finally {
    foreach ($f in @($log, $skips)) { if ([IO.File]::Exists($f)) { [IO.File]::Delete($f) } }
}
