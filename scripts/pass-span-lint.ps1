#Requires -Version 5.1
#
# pass-span-lint.ps1 - refuse a second open `pass_span` on the same render pass (a wgpu validation error on Vulkan).
# The PowerShell twin of scripts/pass-span-lint.py; that header holds the why, and the twins change together.
#
#   powershell -ExecutionPolicy Bypass -File scripts\pass-span-lint.ps1 [path ...]   (default: every crates/**/*.rs)
#   Exit: 0 clean, 1 findings.

$ErrorActionPreference = 'Stop'
$dash = [string][char]0x2014

$OPEN = [regex]'\blet\s+(?:mut\s+)?(\w+)\s*=\s*[\w.]+\.pass_span\(\s*&mut\s+(\w+)\s*,'
$ANY_OPEN = [regex]'\.pass_span\(\s*&mut\s+(\w+)\s*,'
$END = [regex]'\b(\w+)\.end\(\s*&mut\s+(\w+)\s*\)'
$FN = [regex]'^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+\w+'
# str.splitlines()'s line boundaries.
$LINE_BREAK = [regex]"\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]"

# A path as pathlib prints it on Windows: backslashes, no `.` or empty parts.
function Format-PathText([string]$p) {
    $parts = @($p -split '[\\/]' | Where-Object { $_ -ne '' -and $_ -ne '.' })
    $text = $parts -join '\'
    if ($p -match '^[\\/]') { $text = '\' + $text }
    if ($text -eq '') { $text = '.' }
    return $text
}

# pathlib's ordering of Windows paths: part by part, case-folded.
function Compare-PathParts([string]$a, [string]$b) {
    $pa = $a.ToLowerInvariant().Split('\')
    $pb = $b.ToLowerInvariant().Split('\')
    $n = [Math]::Min($pa.Length, $pb.Length)
    for ($i = 0; $i -lt $n; $i++) {
        $c = [string]::CompareOrdinal($pa[$i], $pb[$i])
        if ($c -ne 0) { return $c }
    }
    return $pa.Length - $pb.Length
}

function Sort-Paths([string[]]$paths) {
    $list = New-Object 'System.Collections.Generic.List[string]'
    foreach ($p in $paths) { $list.Add($p) }
    $list.Sort([Comparison[string]] { param($x, $y) Compare-PathParts $x $y })
    return , $list.ToArray()
}

function Invoke-Lint([string]$path) {
    $findings = New-Object 'System.Collections.Generic.List[string]'
    $open = New-Object 'System.Collections.Generic.List[object]'
    $strict = New-Object Text.UTF8Encoding($false, $true)
    $abs = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($path)
    $text = $strict.GetString([IO.File]::ReadAllBytes($abs))
    $lines = $LINE_BREAK.Split($text)
    $count = $lines.Length
    if ($count -gt 0 -and $lines[$count - 1] -eq '') { $count-- }
    for ($n = 0; $n -lt $count; $n++) {
        $lineno = $n + 1
        $raw = $lines[$n]
        $cut = $raw.IndexOf('//')
        $line = $raw
        if ($cut -ge 0) { $line = $raw.Substring(0, $cut) }
        if ($FN.Match($line).Success) { $open.Clear() }
        $guard = $null
        $passName = $null
        $m = $OPEN.Match($line)
        if ($m.Success) {
            $guard = $m.Groups[1].Value
            $passName = $m.Groups[2].Value
        } else {
            $m = $ANY_OPEN.Match($line)
            if ($m.Success) { $passName = $m.Groups[1].Value }
        }
        if ($null -ne $passName) {
            foreach ($o in $open) {
                if ($o[1] -ceq $passName) {
                    $findings.Add(("{0}:{1}: a second pass_span on ``{2}`` while ``{3}`` (line {4}) is still open {5} one pipeline-statistics query at a time" -f
                        $path, $lineno, $passName, $o[0], $o[2], $dash))
                }
            }
            if ($null -ne $guard) { $open.Add(@($guard, $passName, $lineno)) }
        }
        $e = $END.Match($line)
        if ($e.Success) {
            $kept = New-Object 'System.Collections.Generic.List[object]'
            foreach ($o in $open) { if ($o[0] -cne $e.Groups[1].Value) { $kept.Add($o) } }
            $open = $kept
        }
    }
    return , $findings
}

$targets = @($args | ForEach-Object { [string]$_ })
if ($targets.Count -eq 0) { $targets = @('crates') }
$files = New-Object 'System.Collections.Generic.List[string]'
foreach ($a in $targets) {
    $shown = Format-PathText $a
    $absArg = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($a)
    if ([IO.Directory]::Exists($absArg)) {
        $found = New-Object 'System.Collections.Generic.List[string]'
        $full = [IO.Path]::GetFullPath($absArg).TrimEnd('\', '/')
        foreach ($f in [IO.Directory]::EnumerateFiles($full, '*.rs', [IO.SearchOption]::AllDirectories)) {
            $rel = $f.Substring($full.Length + 1)
            if ($shown -eq '.') { $found.Add($rel) } else { $found.Add($shown + '\' + $rel) }
        }
        foreach ($f in (Sort-Paths $found.ToArray())) { $files.Add($f) }
    } else {
        $files.Add($shown)
    }
}

$all = New-Object 'System.Collections.Generic.List[string]'
foreach ($p in $files) {
    foreach ($f in (Invoke-Lint $p)) { $all.Add($f) }
}
foreach ($f in $all) { Write-Output $f }
$verdict = 'clean'
if ($all.Count -gt 0) { $verdict = "$($all.Count) finding(s)" }
Write-Output "pass-span-lint: $verdict $dash $($files.Count) file(s)"
if ($all.Count -gt 0) { exit 1 }
exit 0
