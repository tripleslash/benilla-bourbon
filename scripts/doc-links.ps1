#Requires -Version 5.1
#
# doc-links.ps1 - fail on a doc link whose target is declared nowhere in the workspace.
# The PowerShell twin of scripts/doc-links.py; that header holds the why, and the twins change together.
#
#   powershell -ExecutionPolicy Bypass -File scripts\doc-links.ps1            # the gate
#   powershell -ExecutionPolicy Bypass -File scripts\doc-links.ps1 --report   # + the benign counts

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$dash = [string][char]0x2014

$usage = @'
Fail on a doc link whose target is declared nowhere in the workspace.

It reads rustdoc's unresolved-link warnings and fails on a `crate::`, `super::`, `self::` or
`Self::` path whose leaf name is declared nowhere: the item is gone. Every other warning is a
convention here, not rot (a private module named through its parent, another crate's module
named with `crate::`, a bare name for an unimported bevy type).

    scripts/doc-links.ps1            # the gate: exit 1 on any dead-target link
    scripts/doc-links.ps1 --report   # + the benign counts

The leaf test is a loose declaration grep: a name declared anywhere counts, since what it catches
is a deleted item, not a misspelled path.
'@

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

# Run a program in $root with stdout and stderr captured apart; returns @(exit code, stdout, stderr).
function Invoke-Captured([string]$exe, [string[]]$argList) {
    $utf8 = New-Object Text.UTF8Encoding $false
    $psi = New-Object Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    $psi.Arguments = (@($argList | ForEach-Object { ConvertTo-NativeArg $_ }) -join ' ')
    $psi.UseShellExecute = $false
    $psi.WorkingDirectory = $root
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = $utf8
    $psi.StandardErrorEncoding = $utf8
    $p = [Diagnostics.Process]::Start($psi)
    $outTask = $p.StandardOutput.ReadToEndAsync()
    $errTask = $p.StandardError.ReadToEndAsync()
    $p.WaitForExit()
    $out = $outTask.Result.Replace("`r`n", "`n").Replace("`r", "`n")
    $err = $errTask.Result.Replace("`r`n", "`n").Replace("`r", "`n")
    return @($p.ExitCode, $out, $err)
}

function Get-DeclaredNames {
    $names = New-Object 'System.Collections.Generic.HashSet[string]'
    # `fn` gets its own pattern: in one alternation, `const fn foo` captures "fn" and never indexes `foo`.
    $pats = @(
        [regex]'\bfn\s+([A-Za-z_][A-Za-z0-9_]*)',
        [regex]'\b(?:struct|enum|trait|type|const|static|union|mod)\s+([A-Za-z_][A-Za-z0-9_]*)',
        [regex]'\bmacro_rules!\s+([A-Za-z_][A-Za-z0-9_]*)'
    )
    $utf8 = New-Object Text.UTF8Encoding($false, $false)
    $crates = Join-Path $root 'crates'
    foreach ($rs in [IO.Directory]::EnumerateFiles($crates, '*.rs', [IO.SearchOption]::AllDirectories)) {
        try { $text = $utf8.GetString([IO.File]::ReadAllBytes($rs)) } catch { continue }
        foreach ($pat in $pats) {
            foreach ($m in $pat.Matches($text)) { [void]$names.Add($m.Groups[1].Value) }
        }
    }
    [void]$names.Remove('fn')
    return , $names
}

foreach ($a in $args) {
    if ($a -ceq '-h' -or $a -ceq '--help') {
        Write-Output $usage.Trim()
        exit 0
    }
}
$report = @($args | Where-Object { $_ -ceq '--report' }).Count -gt 0

$r = Invoke-Captured 'cargo' @('doc', '--workspace', '--no-deps', '--document-private-items')
$text = $r[2] + $r[1]
$found = @([regex]::Matches($text, 'warning: unresolved link to `([^`]+)`\n\s*--> ([^\s:]+):(\d+)'))

$names = Get-DeclaredNames
$dead = New-Object Collections.Generic.List[object]
$benign = 0
foreach ($m in $found) {
    $target = $m.Groups[1].Value
    # Only a `crate::`/`super::`/`self::`/`Self::` path asserts a local item.
    if ($target -cnotmatch '^(crate|super|self|Self)::') {
        $benign++
        continue
    }
    $parts = $target -split '::'
    $leaf = ($parts[$parts.Length - 1] -split '\(')[0].Trim()
    if (-not $leaf -or $leaf -cnotmatch '^[A-Za-z_][A-Za-z0-9_]*$') {
        $benign++
        continue
    }
    if ($names.Contains($leaf)) { $benign++ }
    else { $dead.Add(@($target, $m.Groups[2].Value, $m.Groups[3].Value)) }
}

if ($report) {
    Write-Output "doc-links: $($found.Count) unresolved-link warnings"
    Write-Output "doc-links: $benign name something that still exists (see this file's header)"
}

if ($dead.Count -eq 0) {
    Write-Output "doc-links ok: no doc link points at a deleted item ($($found.Count) benign)"
    exit 0
}

Write-Output "doc-links FAILED: $($dead.Count) doc link(s) point at something that no longer exists:`n"
foreach ($d in $dead) {
    Write-Output "  $($d[1]):$($d[2])`n      [``$($d[0])``]"
}
Write-Output ("`nEach names an item declared nowhere in the workspace. Repoint it at whatever replaced`n" +
    "the thing, or drop the brackets $dash a dangling link promises navigation that is not there.")
exit 1
