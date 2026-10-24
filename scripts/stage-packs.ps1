#Requires -Version 5.1
#
# stage-packs.ps1 - assemble web/site/, the directory a static host serves.
# The PowerShell twin of scripts/stage-packs.sh; that header holds the why, and the twins change together.
#
#   powershell -ExecutionPolicy Bypass -File scripts\stage-packs.ps1
#
#   STAGE_PACKS_LINK=hardlink   link every pack individually instead of linking the directory.
#   LOOSE_DATA, LOOSE_DATA_TURTLE   installs whose loose *.html files are staged into the containers.

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

$PACKS = 'web/packs'
$PACKS_HD = 'web/packs-hd'
$PACKS_TURTLE = 'web/packs-turtle'
$PACKS_REFORGED = 'web/packs-reforged'
$SITE = 'web/site'
$WASM_PART_MAX = 24 * 1024 * 1024

# Byte-transparent: every byte maps to one char and back, so a rewrite never re-encodes the rest.
$latin1 = [Text.Encoding]::GetEncoding(28591)

function Resolve-Repo([string]$path) {
    return [IO.Path]::GetFullPath([IO.Path]::Combine($root, $path))
}

function Write-Err([string]$text) {
    [Console]::Error.WriteLine($text)
}

# What a bash `*` glob would match: no dot-names.
function Test-Visible($item) {
    return -not $item.Name.StartsWith('.')
}

# rm -rf that removes a junction or symlink itself and never walks into its target.
function Remove-Entry($item) {
    $isLink = [bool]($item.Attributes -band [IO.FileAttributes]::ReparsePoint)
    if ($item.PSIsContainer) {
        if (-not $isLink) {
            foreach ($child in @(Get-ChildItem -LiteralPath $item.FullName -Force)) { Remove-Entry $child }
        }
        [IO.Directory]::Delete($item.FullName, $false)
    } else {
        if ($item.Attributes -band [IO.FileAttributes]::ReadOnly) {
            $item.Attributes = $item.Attributes -bxor [IO.FileAttributes]::ReadOnly
        }
        [IO.File]::Delete($item.FullName)
    }
}

# cp -r src/. dst/
function Copy-Tree([string]$src, [string]$dst) {
    [void][IO.Directory]::CreateDirectory($dst)
    foreach ($child in @(Get-ChildItem -LiteralPath $src -Force)) {
        $target = [IO.Path]::Combine($dst, $child.Name)
        if ($child.PSIsContainer) { Copy-Tree $child.FullName $target }
        else { [IO.File]::Copy($child.FullName, $target, $true) }
    }
}

# cp -lr src/. dst/
function Link-Tree([string]$src, [string]$dst) {
    [void][IO.Directory]::CreateDirectory($dst)
    foreach ($child in @(Get-ChildItem -LiteralPath $src -Force)) {
        $target = [IO.Path]::Combine($dst, $child.Name)
        if ($child.PSIsContainer) { Link-Tree $child.FullName $target }
        else { New-Item -ItemType HardLink -Path $target -Value $child.FullName | Out-Null }
    }
}

# First candidate that exists and is non-empty; staging outranks dist.
$dist = ''
foreach ($candidate in @($env:TRUNK_STAGING_DIR, 'web/dist/.stage', 'web/dist')) {
    if ([string]::IsNullOrEmpty($candidate)) { continue }
    $abs = Resolve-Repo $candidate
    if ([IO.Directory]::Exists($abs) -and [IO.Directory]::EnumerateFileSystemEntries($abs).GetEnumerator().MoveNext()) {
        $dist = $candidate
        break
    }
}
if (-not $dist) {
    Write-Err 'no build found in web/dist or web/dist/.stage - run `trunk build --release` first'
    exit 1
}
Write-Host "staging from $dist"
$distAbs = Resolve-Repo $dist

# Workers have no import map: point the glue's bare specifier at the file, before anything links it.
$scripts = @(Get-ChildItem -LiteralPath $distAbs -File -Force |
    Where-Object { (Test-Visible $_) -and ($_.Name -cmatch '\.js$') })
foreach ($js in $scripts) {
    $text = $latin1.GetString([IO.File]::ReadAllBytes($js.FullName))
    if ($text.IndexOf('wasi_snapshot_preview1', [StringComparison]::Ordinal) -lt 0) { continue }
    $text = $text.Replace('from "wasi_snapshot_preview1"', 'from "./wasi_stubs.js"')
    $text = $text.Replace("from 'wasi_snapshot_preview1'", "from './wasi_stubs.js'")
    [IO.File]::WriteAllBytes($js.FullName, $latin1.GetBytes($text))
    Write-Host "  rewrote the wasi_snapshot_preview1 specifier in $($js.Name) (workers have no import map)"
}

if (-not [IO.Directory]::Exists((Resolve-Repo $PACKS))) {
    Write-Err "warning: no $PACKS - staging the client with no game data (benilla-pack build)"
}
if (-not [IO.Directory]::Exists((Resolve-Repo $PACKS_HD))) {
    Write-Err "note: no $PACKS_HD - staging without the updated-character-model layer"
}
if (-not [IO.Directory]::Exists((Resolve-Repo $PACKS_TURTLE))) {
    Write-Err "note: no $PACKS_TURTLE - staging without the Turtle WoW 1.18.1 layer"
}
if (-not [IO.Directory]::Exists((Resolve-Repo $PACKS_REFORGED))) {
    Write-Err "note: no $PACKS_REFORGED - staging without the reforged art layer"
}

# The install's loose *.html files, lowercased into <container>/loose/; the installs are only read.
function Stage-Loose([string]$src, [string]$container) {
    if ([string]::IsNullOrEmpty($src)) { return }
    $srcAbs = Resolve-Repo $src
    $dstAbs = Resolve-Repo $container
    if (-not ([IO.Directory]::Exists($srcAbs) -and [IO.Directory]::Exists($dstAbs))) { return }
    $loose = [IO.Path]::Combine($dstAbs, 'loose')
    [void][IO.Directory]::CreateDirectory($loose)
    $pages = @(Get-ChildItem -LiteralPath $srcAbs -File -Force |
        Where-Object { (Test-Visible $_) -and ($_.Name -cmatch '\.html$') })
    foreach ($f in $pages) {
        [IO.File]::Copy($f.FullName, [IO.Path]::Combine($loose, $f.Name.ToLowerInvariant()), $true)
    }
    $count = @(Get-ChildItem -LiteralPath $loose -Force | Where-Object { Test-Visible $_ }).Count
    Write-Host "  staged $count loose file(s) from $src into $container/loose"
}
Stage-Loose $env:LOOSE_DATA $PACKS
Stage-Loose $env:LOOSE_DATA_TURTLE $PACKS_TURTLE

$siteAbs = Resolve-Repo $SITE
[void][IO.Directory]::CreateDirectory($siteAbs)
# Everything except the container links.
$keep = @('data', 'data-hd', 'data-turtle', 'data-reforged')
foreach ($item in @(Get-ChildItem -LiteralPath $siteAbs -Force)) {
    if ($keep -ccontains $item.Name) { continue }
    Remove-Entry $item
}

# Hardlink the build; copy where the filesystem refuses a link.
foreach ($item in @(Get-ChildItem -LiteralPath $distAbs -Force | Where-Object { Test-Visible $_ })) {
    $target = [IO.Path]::Combine($siteAbs, $item.Name)
    if ($item.PSIsContainer) {
        Copy-Tree $item.FullName $target
        continue
    }
    try {
        New-Item -ItemType HardLink -Path $target -Value $item.FullName | Out-Null
    } catch {
        [IO.File]::Copy($item.FullName, $target, $true)
    }
}

# The bundled addons as loose files, with index.txt listing them in bytewise order.
$addonsAbs = Resolve-Repo 'third_party/addons'
if ([IO.Directory]::Exists($addonsAbs)) {
    $addonsOut = [IO.Path]::Combine($siteAbs, 'addons')
    Copy-Tree $addonsAbs $addonsOut
    $utf8 = New-Object Text.UTF8Encoding $false
    $keys = New-Object 'System.Collections.Generic.List[string]'
    foreach ($f in [IO.Directory]::EnumerateFiles($addonsOut, '*', [IO.SearchOption]::AllDirectories)) {
        if ([IO.Path]::GetFileName($f) -ceq 'index.txt') { continue }
        $rel = $f.Substring($addonsOut.Length + 1).Replace('\', '/')
        # Latin-1 of the UTF-8 bytes, so an ordinal sort is the bytewise (LC_ALL=C) sort.
        $keys.Add($latin1.GetString($utf8.GetBytes($rel)))
    }
    $sorted = $keys.ToArray()
    [Array]::Sort($sorted, [StringComparer]::Ordinal)
    $index = ''
    if ($sorted.Length -gt 0) { $index = [string]::Join("`n", $sorted) + "`n" }
    [IO.File]::WriteAllBytes([IO.Path]::Combine($addonsOut, 'index.txt'), $latin1.GetBytes($index))
    Write-Host "staged $($sorted.Length) bundled addon files"
}

function Read-Full($stream, [byte[]]$buf) {
    $total = 0
    while ($total -lt $buf.Length) {
        $n = $stream.Read($buf, $total, $buf.Length - $total)
        if ($n -le 0) { break }
        $total += $n
    }
    return $total
}

# Split the wasm module into 24 MiB *.wasm parts plus wenilla_bg.parts.json.
$wasmAbs = [IO.Path]::Combine($siteAbs, 'wenilla_bg.wasm')
if ([IO.File]::Exists($wasmAbs)) {
    $bytes = (New-Object IO.FileInfo $wasmAbs).Length
    $stem = $wasmAbs.Substring(0, $wasmAbs.Length - 5)
    $buf = New-Object byte[] $WASM_PART_MAX
    $in = [IO.File]::OpenRead($wasmAbs)
    try {
        $i = 0
        while ($true) {
            $n = Read-Full $in $buf
            if ($n -le 0) { break }
            $out = [IO.File]::Create(('{0}.part{1:d2}.wasm' -f $stem, $i))
            try { $out.Write($buf, 0, $n) } finally { $out.Dispose() }
            $i++
        }
    } finally {
        $in.Dispose()
    }
    # The link, not the build: web/dist keeps its inode.
    [IO.File]::Delete($wasmAbs)
    [string[]]$parts = @(Get-ChildItem -LiteralPath $siteAbs -File -Force |
        Where-Object { $_.Name -clike 'wenilla_bg.part*.wasm' } | ForEach-Object { $_.Name })
    [Array]::Sort($parts, [StringComparer]::Ordinal)
    $quoted = @($parts | ForEach-Object { '"' + $_ + '"' })
    $json = '{"bytes":' + $bytes + ',"parts":[' + ($quoted -join ',') + ']}' + "`n"
    [IO.File]::WriteAllText([IO.Path]::Combine($siteAbs, 'wenilla_bg.parts.json'), $json, (New-Object Text.UTF8Encoding $false))
    Write-Host "split wenilla_bg.wasm ($bytes B) into $($parts.Length) parts"
}

# Nothing at the top of the site may exceed the cap; the packs are capped by the packer.
$oversize = @(Get-ChildItem -LiteralPath $siteAbs -File -Force | Where-Object { $_.Length -gt $WASM_PART_MAX })
if ($oversize.Count -gt 0) {
    Write-Err "error: asset over a static host's 25 MiB limit (cap here is 24 MiB):"
    foreach ($f in $oversize) { Write-Err "$SITE/$($f.Name)" }
    exit 1
}

# $src = the container directory, $name = the name it takes under the site.
function Link-Data([string]$src, [string]$name) {
    $srcAbs = Resolve-Repo $src
    $dst = [IO.Path]::Combine($siteAbs, $name)
    if ($env:STAGE_PACKS_LINK -ceq 'hardlink') {
        if (Test-Path -LiteralPath $dst) { Remove-Entry (Get-Item -LiteralPath $dst -Force) }
        Link-Tree $srcAbs $dst
        return
    }
    if (Test-Path -LiteralPath $dst) { return }
    if ($env:OS -eq 'Windows_NT') {
        # A junction needs neither administrator rights nor Developer Mode; a symlink does.
        New-Item -ItemType Junction -Path $dst -Value $srcAbs | Out-Null
    } else {
        $relative = '../' + [IO.Path]::GetFileName($srcAbs)
        if ([IO.Directory].GetMethod('CreateSymbolicLink', [type[]]@([string], [string]))) {
            [void][IO.Directory]::CreateSymbolicLink($dst, $relative)
        } else {
            New-Item -ItemType SymbolicLink -Path $dst -Value $relative | Out-Null
        }
    }
}
if ([IO.Directory]::Exists((Resolve-Repo $PACKS))) { Link-Data $PACKS 'data' }
if ([IO.Directory]::Exists((Resolve-Repo $PACKS_HD))) { Link-Data $PACKS_HD 'data-hd' }
if ([IO.Directory]::Exists((Resolve-Repo $PACKS_TURTLE))) { Link-Data $PACKS_TURTLE 'data-turtle' }
if ([IO.Directory]::Exists((Resolve-Repo $PACKS_REFORGED))) { Link-Data $PACKS_REFORGED 'data-reforged' }

# Counted off the container trees, not through the links.
$objects = @(Get-ChildItem -LiteralPath $siteAbs -File -Force).Count
foreach ($tree in @($PACKS, $PACKS_HD, $PACKS_TURTLE, $PACKS_REFORGED)) {
    $treeAbs = Resolve-Repo $tree
    if (-not [IO.Directory]::Exists($treeAbs)) { continue }
    foreach ($f in [IO.Directory]::EnumerateFiles($treeAbs, '*', [IO.SearchOption]::AllDirectories)) { $objects++ }
}
Write-Host "assets to upload: $objects (a typical static host's limit: 20000)"
if ($objects -gt 20000) {
    Write-Err 'error: over a static host''s 20,000-asset limit - drop a layer or rebuild with fewer objects'
    exit 1
}

Write-Host "staged ${SITE}:"
Get-ChildItem -LiteralPath $siteAbs -Force | Out-Host
exit 0
