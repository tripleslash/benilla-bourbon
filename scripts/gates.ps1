#Requires -Version 5.1
#
# gates.ps1 - the land gate chain: every gate in order, stopping at the first failure.
# The PowerShell twin of scripts/gates.sh; that header holds the why, and the twins change together.
# Runs doc-links.ps1 and pass-span-lint.ps1, and memoizes on this file's own hash.
#
#   powershell -ExecutionPolicy Bypass -File scripts\gates.ps1
#
# GATES_FORCE=1 runs the chain anyway, WOW_DATA picks the install (set and empty: none) and TMPDIR
# holds the scratch files. BENILLA_RE names the benilla-re checkout the reference gate compares
# `reference/` with; that gate and message-catalog run upstream's Python scripts through `python`.

$ErrorActionPreference = 'Stop'
$dash = [string][char]0x2014
$times = [string][char]0x00D7
$utf8 = New-Object Text.UTF8Encoding $false
$clock = [Diagnostics.Stopwatch]::StartNew()
$psExe = [Diagnostics.Process]::GetCurrentProcess().MainModule.FileName

function Get-Seconds { return [int][Math]::Floor($clock.Elapsed.TotalSeconds) }

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

# Gate the checkout the location is in, not this file's: outside a checkout of this repo, gate this file's own.
$here = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$code = Invoke-Native 'git' @('rev-parse', '--show-toplevel') -Capture
$root = $script:NativeOut
if ($code -ne 0) { $root = '' }
if (-not $root -or -not [IO.File]::Exists([IO.Path]::Combine($root, 'scripts\gates.ps1'))) { $root = $here }
Set-Location -LiteralPath $root
[Environment]::CurrentDirectory = (Get-Location).ProviderPath
Write-Output "gating: $root"

$stamp = 'target/.gates-green'
$corpus = ''
if ([IO.Directory]::Exists([IO.Path]::Combine($root, 'wow-addons-vanilla'))) { $corpus = '+corpus' }

function Get-TreeKey {
    $tmpidx = New-TempFile 'benilla-gates-idx'
    [IO.File]::Delete($tmpidx)
    $t = ''
    try {
        $idx = @{ GIT_INDEX_FILE = $tmpidx }
        $rc = Invoke-Native 'git' @('read-tree', 'HEAD') -EnvVars $idx -Capture
        $t += $script:NativeOut
        if ($rc -eq 0) {
            $rc = Invoke-Native 'git' @('add', '-A', '.') -EnvVars $idx -Capture
            $t += $script:NativeOut
            if ($rc -eq 0) {
                [void](Invoke-Native 'git' @('write-tree') -EnvVars $idx -Capture)
                $t += $script:NativeOut
            }
        }
    } finally {
        if ([IO.File]::Exists($tmpidx)) { [IO.File]::Delete($tmpidx) }
    }
    if (-not $t) { return '' }
    [void](Invoke-Native 'rustc' @('-V') -Capture)
    $rustc = $script:NativeOut
    $sha = ''
    $self = [IO.Path]::Combine($root, 'scripts\gates.ps1')
    if ([IO.File]::Exists($self)) {
        $sha1 = [Security.Cryptography.SHA1]::Create()
        try { $hash = $sha1.ComputeHash([IO.File]::ReadAllBytes($self)) } finally { $sha1.Dispose() }
        $sha = -join ($hash | ForEach-Object { $_.ToString('x2') })
    }
    return '{0}|{1}|{2}' -f $t, $rustc, $sha
}

function Resolve-Wow {
    [void](Invoke-Native 'cargo' @('run', '-q', '-p', 'benilla-formats', '--example', 'where') -Capture)
    return $script:NativeOut
}

# A delta only under docs/, in top-level *.md files or in the issue and PR templates keeps the stamped verdict.
function Test-DocsOnlyDelta([string]$a, [string]$b) {
    if ((Invoke-Native 'git' @('cat-file', '-e', "$a^{tree}") -Quiet) -ne 0) { return $false }
    if ((Invoke-Native 'git' @('cat-file', '-e', "$b^{tree}") -Quiet) -ne 0) { return $false }
    [void](Invoke-Native 'git' @('diff-tree', '-r', '--name-only', $a, $b) -Capture)
    foreach ($line in $script:NativeOut.Split([char]10)) {
        if ($line -ne '' -and $line -cnotmatch '^(docs/|[^/]*\.md$|\.github/(ISSUE_TEMPLATE/|pull_request_template\.md$))') { return $false }
    }
    return $true
}

function Get-UtcStamp { return [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ', [Globalization.CultureInfo]::InvariantCulture) }

function Get-ShortHead {
    $rc = Invoke-Native 'git' @('rev-parse', '--short', 'HEAD') -Capture
    if ($rc -ne 0) { return '-' }
    return $script:NativeOut
}

function Add-Line([string]$path, [string]$line) {
    try { [IO.File]::AppendAllText([IO.Path]::Combine($root, $path), $line + "`n", $utf8) } catch { }
}

$smokeLine = "  (a clean run is the fourth gate: scripts/smoke.ps1 $dash live logout/re-login round trip)"

$keyStart = Get-TreeKey
if ($env:GATES_FORCE -ne '1' -and $keyStart -and [IO.File]::Exists([IO.Path]::Combine($root, $stamp))) {
    $old = $utf8.GetString([IO.File]::ReadAllBytes([IO.Path]::Combine($root, $stamp))).TrimEnd([char]10)
    $cut = $old.LastIndexOf('|')
    $oldKey = $old
    $oldWow = $old
    if ($cut -ge 0) { $oldKey = $old.Substring(0, $cut); $oldWow = $old.Substring($cut + 1) }
    $memo = ''
    if ($oldKey -ceq $keyStart) {
        $memo = 'this exact tree already passed'
    } else {
        $oldRest = $oldKey.Substring($oldKey.IndexOf('|') + 1)
        $newRest = $keyStart.Substring($keyStart.IndexOf('|') + 1)
        if ($oldRest -ceq $newRest -and (Test-DocsOnlyDelta $oldKey.Split('|')[0] $keyStart.Split('|')[0])) {
            $memo = 'this tree differs from one that passed only under docs/ or in top-level *.md, which no gate reads'
        }
    }
    if ($memo -and $oldWow -ceq ((Resolve-Wow) + $corpus)) {
        Write-Output "ALL GATES GREEN (memoized $dash $memo; GATES_FORCE=1 re-runs)"
        if ([IO.Directory]::Exists([IO.Path]::Combine($root, 'target'))) {
            Add-Line 'target/.gates-timing' ("{0}`t{1}`tchain`t0`tmemo" -f (Get-UtcStamp), (Get-ShortHead))
        }
        Write-Output $smokeLine
        exit 0
    }
}

$log = New-TempFile 'benilla-gates'
$skips = New-TempFile 'benilla-gates-skips'
try {
    # Each data-gated skip the tests logged, grouped, after a test rung; then the file is emptied.
    function Write-Skips([string]$rung) {
        $lines = Read-Lines $skips
        if ($lines.Count -eq 0) { return }
        Write-Output "  ${rung}: $($lines.Count) data-gated tests SKIPPED on this machine $dash"
        $counts = New-Object 'System.Collections.Generic.Dictionary[string,int]'
        $order = New-Object 'System.Collections.Generic.List[string]'
        foreach ($l in $lines) {
            if ($counts.ContainsKey($l)) { $counts[$l]++ } else { $counts[$l] = 1; $order.Add($l) }
        }
        # sort | uniq -c | sort -rn: the most frequent first, ties in reverse byte order.
        $rows = New-Object 'System.Collections.Generic.List[string]'
        foreach ($l in $order) { $rows.Add($l) }
        $rows.Sort([Comparison[string]] {
                param($x, $y)
                if ($counts[$x] -ne $counts[$y]) { return $counts[$y] - $counts[$x] }
                return [string]::CompareOrdinal($y, $x)
            })
        foreach ($r in $rows) { Write-Output "    $($counts[$r]) $times $r" }
        Write-Output '    (they run where the data is: docs/CONTRIBUTING.md, "Setting up")'
        [IO.File]::WriteAllBytes($skips, [byte[]]@())
    }

    $timing = 'target/.gates-timing'
    $chainT0 = Get-Seconds
    $treeSha = Get-ShortHead
    function Add-Timing([string]$gate, [int]$secs, [string]$verdict) {
        if (-not [IO.Directory]::Exists([IO.Path]::Combine($root, 'target'))) { return }
        Add-Line $timing ("{0}`t{1}`t{2}`t{3}`t{4}" -f (Get-UtcStamp), $treeSha, $gate, $secs, $verdict)
    }

    function Invoke-Gate([string]$name, [string]$exe, [string[]]$argList, [hashtable]$envMods = @{}) {
        $t0 = Get-Seconds
        $rc = Invoke-Native $exe $argList -EnvVars $envMods -LogPath $log
        $secs = (Get-Seconds) - $t0
        if ($rc -ne 0) {
            Write-Output "GATE FAILED: $name ($secs s)"
            Add-Timing $name $secs 'FAILED'
            # The failing test or error first: a test gate's tail is the doc-test noise after it.
            $lines = Read-Lines $log
            @($lines | Where-Object { $_ -cmatch 'panicked at|^---- .* ----$|^test .* FAILED$|^error(\[E[0-9]+\])?:' }) |
                Select-Object -First 40 | ForEach-Object { Write-Output $_ }
            $lines | Select-Object -Last 30 | ForEach-Object { Write-Output $_ }
            exit 1
        }
        Write-Output "gate ok: $name ($secs s)"
        Add-Timing $name $secs 'ok'
    }

    function Get-ScriptArgs([string]$name) {
        return @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', [IO.Path]::Combine($root, "scripts\$name"))
    }

    Invoke-Gate 'fmt' 'cargo' @('fmt', '--all', '--', '--check')
    Invoke-Gate 'clippy' 'cargo' @('clippy', '--workspace', '--all-targets', '--', '-D', 'warnings')

    $wowData = Resolve-Wow
    $requireData = ''
    if ($wowData -and $corpus) {
        $requireData = '1'
    } else {
        $shownWow = $wowData
        if (-not $shownWow) { $shownWow = 'none' }
        $hasCorpus = 'no'
        if ([IO.Directory]::Exists([IO.Path]::Combine($root, 'wow-addons-vanilla'))) { $hasCorpus = 'yes' }
        Write-Output ("gates: NOTE $dash install or addon corpus not found here (install='$shownWow', " +
            "corpus=$hasCorpus): data-gated tests skip silently on this machine")
    }

    if ($env:BENILLA_RE -and -not [IO.Directory]::Exists([IO.Path]::Combine($env:BENILLA_RE, 'tables'))) {
        Write-Output "GATE FAILED: reference (BENILLA_RE=$($env:BENILLA_RE) holds no benilla-re checkout)"
        exit 1
    }
    $candidates = @($env:BENILLA_RE, [IO.Path]::Combine($root, '..', 'benilla-re'))
    if ((Invoke-Native 'git' @('rev-parse', '--git-common-dir') -Capture) -eq 0 -and $script:NativeOut) {
        $primary = [IO.Path]::GetDirectoryName([IO.Path]::GetFullPath([IO.Path]::Combine($root, $script:NativeOut)))
        $candidates += [IO.Path]::Combine($primary, '..', 'benilla-re')
    }
    $reCheckout = $candidates | Where-Object { $_ -and [IO.Directory]::Exists([IO.Path]::Combine($_, 'tables')) } | Select-Object -First 1
    if ($reCheckout) {
        Invoke-Gate 'reference' 'python' @([IO.Path]::Combine($root, 'scripts', 'sync-reference.py'), $reCheckout, '--check')
    } elseif ($requireData -or $env:BENILLA_REQUIRE_DATA -eq '1') {
        Write-Output 'GATE FAILED: reference (no benilla-re checkout: set BENILLA_RE, or clone it beside the repo)'
        exit 1
    } else {
        Write-Output 'gate SKIPPED: reference (no benilla-re checkout found: BENILLA_RE, ../benilla-re)'
        Add-Timing 'reference' 0 'SKIPPED'
    }
    $testEnv = @{ BENILLA_SKIP_LOG = $skips }
    if ($requireData) { $testEnv['BENILLA_REQUIRE_DATA'] = '1' }
    Invoke-Gate 'test' 'cargo' @('test', '--no-fail-fast', '--workspace') $testEnv
    Write-Skips 'test'

    Invoke-Gate 'test-no-install' 'cargo' @('test', '--no-fail-fast', '--workspace') @{ BENILLA_REQUIRE_DATA = $null; WOW_DATA = '' }

    Invoke-Gate 'doc-links' $psExe (Get-ScriptArgs 'doc-links.ps1')
    Invoke-Gate 'message-catalog' 'python' @([IO.Path]::Combine($root, 'scripts', 'gen-message-catalog.py'), '--check')
    Invoke-Gate 'pass-span-lint' $psExe (Get-ScriptArgs 'pass-span-lint.ps1')

    $playerEnv = @{ BENILLA_SKIP_LOG = $skips }
    if ($wowData) { $playerEnv['WOW_DATA'] = $wowData }
    if ($corpus) { $playerEnv['BENILLA_ADDON_CORPUS'] = [IO.Path]::Combine($root, 'wow-addons-vanilla') }
    if ($requireData) { $playerEnv['BENILLA_REQUIRE_DATA'] = '1' }
    Invoke-Gate 'player-tests' 'cargo' @('test', '-p', 'benilla-formats', '-p', 'benilla-app', '--no-default-features', '--lib') $playerEnv
    Write-Skips 'player-tests'

    if ((Invoke-Native 'cargo' @('run', '-q', '-p', 'benilla-formats', '--example', 'where') -Quiet) -eq 0) {
        Invoke-Gate 'enforcer' 'cargo' @('run', '-q', '-p', 'benilla-worldview') @{ WOW_WORLDVIEW_CHECK = '10' }
    } else {
        Write-Output "gate SKIPPED: enforcer (no WoW install found $dash the engine boot check needs one;"
        Write-Output '             `cargo run -p benilla-formats --example where` says where it looked)'
        Add-Timing 'enforcer' 0 'SKIPPED'
    }

    Invoke-Gate 'enforcer-no-install' 'cargo' @('run', '-q', '-p', 'benilla-worldview') @{ WOW_DATA = ''; WOW_WORLDVIEW_CHECK = '5' }

    $chainSecs = (Get-Seconds) - $chainT0
    Write-Output "ALL GATES GREEN ($chainSecs s; per-gate seconds in $timing)"
    Add-Timing 'chain' $chainSecs 'ok'

    # Stamp only if the tree still hashes as it did at the start.
    $keyEnd = Get-TreeKey
    if ($keyEnd -and $keyEnd -ceq $keyStart -and [IO.Directory]::Exists([IO.Path]::Combine($root, 'target'))) {
        try {
            [IO.File]::WriteAllText([IO.Path]::Combine($root, $stamp), ('{0}|{1}{2}' -f $keyEnd, $wowData, $corpus) + "`n", $utf8)
        } catch { }
    }

    Write-Output $smokeLine
    exit 0
} finally {
    foreach ($f in @($log, $skips)) { if ([IO.File]::Exists($f)) { [IO.File]::Delete($f) } }
}
