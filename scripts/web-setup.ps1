#Requires -Version 5.1
#
# web-setup.ps1 - one-time toolchain for the browser build (no admin rights, nothing system-wide).
# The PowerShell twin of scripts/web-setup.sh; that header holds the why, and the twins change together.
#
#   powershell -ExecutionPolicy Bypass -File scripts\web-setup.ps1
#
# Re-runnable; each step is skipped when already done. WASI_SDK_VERSION and BINARYEN_VERSION
# override the releases.

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

function Default-Env([string]$name, [string]$fallback) {
    $value = [Environment]::GetEnvironmentVariable($name)
    if ([string]::IsNullOrEmpty($value)) { return $fallback }
    return $value
}

function Resolve-Repo([string]$path) {
    return [IO.Path]::GetFullPath([IO.Path]::Combine($root, $path))
}

$wasiVersion = Default-Env 'WASI_SDK_VERSION' '27'
$wasiDir = 'tools/wasi-sdk'

# The host triple in the naming schemes wasi-sdk and binaryen both use.
$plat = ''
$bplat = ''
$hostName = ''
$tar = 'tar'
if ($env:OS -eq 'Windows_NT') {
    $plat = 'x86_64-windows'
    $bplat = 'x86_64-windows'
    $hostName = 'Windows'
    # The bsdtar Windows ships; a GNU tar earlier on PATH would read C:\ as a remote host.
    $systemTar = [IO.Path]::Combine($env:SystemRoot, 'System32\tar.exe')
    if ([IO.File]::Exists($systemTar)) { $tar = $systemTar }
} else {
    $ri = [System.Runtime.InteropServices.RuntimeInformation]
    $arch = [string]$ri::OSArchitecture
    $onLinux = $ri::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Linux)
    $onMac = $ri::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::OSX)
    $hostName = '{0}-{1}' -f $ri::OSDescription, $arch
    if ($onLinux -and $arch -eq 'X64') { $plat = 'x86_64-linux'; $bplat = 'x86_64-linux' }
    elseif ($onLinux -and $arch -eq 'Arm64') { $plat = 'arm64-linux'; $bplat = 'aarch64-linux' }
    elseif ($onMac -and $arch -eq 'Arm64') { $plat = 'arm64-macos'; $bplat = 'arm64-macos' }
    elseif ($onMac -and $arch -eq 'X64') { $plat = 'x86_64-macos'; $bplat = 'x86_64-macos' }
}

# curl | tar xz --strip-components=1 -C $dest
function Get-Tarball([string]$url, [string]$dest) {
    [void][IO.Directory]::CreateDirectory($dest)
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    $tmp = [IO.Path]::GetTempFileName()
    $rc = 0
    try {
        $wc = New-Object Net.WebClient
        try { $wc.DownloadFile($url, $tmp) } finally { $wc.Dispose() }
        & $script:tar -xzf $tmp --strip-components=1 -C $dest
        $rc = $LASTEXITCODE
    } finally {
        [IO.File]::Delete($tmp)
    }
    if ($rc -ne 0) { exit $rc }
}

Write-Host '== rustup target'
rustup target add wasm32-unknown-unknown
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host '== trunk'
if (Get-Command trunk -ErrorAction SilentlyContinue) {
    $lines = @(trunk --version)
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $version = ''
    if ($lines.Count -gt 0) {
        $fields = @(([string]$lines[0]).Trim() -split '\s+')
        if ($fields.Count -gt 1) { $version = $fields[1] }
    }
    Write-Host "trunk $version already installed"
} else {
    cargo install trunk --locked
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "== wasi-sdk $wasiVersion"
if ([IO.File]::Exists((Resolve-Repo "$wasiDir/share/wasi-sysroot/lib/wasm32-wasip1/libsetjmp.a"))) {
    Write-Host "already at $wasiDir"
} elseif (-not $plat) {
    [Console]::Error.WriteLine("unsupported host $hostName; set WASI_SDK to an unpacked SDK instead")
    exit 1
} else {
    $url = "https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-${wasiVersion}/wasi-sdk-${wasiVersion}.0-${plat}.tar.gz"
    [void][IO.Directory]::CreateDirectory((Resolve-Repo $wasiDir))
    Write-Host "downloading $url"
    Get-Tarball $url (Resolve-Repo $wasiDir)
}

Write-Host '== binaryen (Trunk fetches its own wasm-opt; this is for invoking it by hand)'
$binaryenVersion = Default-Env 'BINARYEN_VERSION' '130'
if ([IO.File]::Exists((Resolve-Repo 'tools/binaryen/bin/wasm-opt')) -or [IO.File]::Exists((Resolve-Repo 'tools/binaryen/bin/wasm-opt.exe'))) {
    Write-Host 'already at tools/binaryen'
} elseif ($bplat) {
    $url = "https://github.com/WebAssembly/binaryen/releases/download/version_${binaryenVersion}/binaryen-version_${binaryenVersion}-${bplat}.tar.gz"
    Get-Tarball $url (Resolve-Repo 'tools/binaryen')
} else {
    Write-Host '  no binaryen release for this host - Trunk''s own wasm-opt still runs'
}

Write-Host @'

ready. The build is now:

  $env:WASI_SDK = "$PWD\tools\wasi-sdk"; trunk build --release      # -> web/dist
  powershell -ExecutionPolicy Bypass -File scripts\stage-packs.ps1    # -> web/site

or, for the build with worker threads (nightly + -Zbuild-std + shared memory; same output dirs,
and it verifies the emitted module really came out threaded):

  powershell -ExecutionPolicy Bypass -File scripts\web-build-threads.ps1

and the game data is a separate, far slower, one-time job:

  cargo run --release -p benilla-packer --bin benilla-pack -- \
      build --data "/path/to/WoW 1.12.1/Data" --out web/packs

Serve both locally with:

  cargo run --release -p wenilla-host -- --www web/dist --packs web/packs
  cargo run --release -p wenilla-host -- --www web/dist --packs web/packs --no-ranges  # as a static host behaves
'@
exit 0
