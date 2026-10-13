#Requires -Version 5.1
#
# web-build-threads.ps1 - the browser build WITH THREADS, for Windows PowerShell 5.1 and later.
# The PowerShell twin of scripts/web-build-threads.sh; that header holds the why, and the twins change together.
#
#   powershell -ExecutionPolicy Bypass -File scripts\web-build-threads.ps1
#
# Overrides, as in the bash script: BENILLA_NIGHTLY, BENILLA_MAX_MEMORY, WASI_SDK,
# CARGO_TARGET_DIR, RUSTFLAGS (prepended), BENILLA_KEEP_NAMES=1.

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

function Default-Env([string]$name, [string]$fallback) {
    $value = [Environment]::GetEnvironmentVariable($name)
    if ([string]::IsNullOrEmpty($value)) { return $fallback }
    return $value
}

$env:RUSTUP_TOOLCHAIN = Default-Env 'BENILLA_NIGHTLY' 'nightly'
$env:CARGO_UNSTABLE_BUILD_STD = 'std,panic_abort'
$env:WASI_SDK = Default-Env 'WASI_SDK' (Join-Path $root 'tools\wasi-sdk')
$env:CARGO_TARGET_DIR = Default-Env 'CARGO_TARGET_DIR' (Join-Path $root 'target\threads')

$maxMemory = Default-Env 'BENILLA_MAX_MEMORY' '4294967296'
# Shared, imported and exported memory plus the five exports wasm-bindgen's threading transform
# reads; the bash script's comments carry the measurements behind each.
$threadFlags = @(
    '-Ctarget-feature=+atomics,+bulk-memory,+mutable-globals',
    '-Clink-arg=--shared-memory',
    '-Clink-arg=--import-memory',
    '-Clink-arg=--export-memory',
    "-Clink-arg=--max-memory=$maxMemory",
    '-Clink-arg=--export=__heap_base',
    '-Clink-arg=--export=__wasm_init_tls',
    '-Clink-arg=--export=__tls_base',
    '-Clink-arg=--export=__tls_size',
    '-Clink-arg=--export=__tls_align'
) -join ' '
$env:RUSTFLAGS = "$env:RUSTFLAGS $threadFlags"

Write-Host '== toolchain'
rustc -vV | ForEach-Object { Write-Host "   $_" }
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "   RUSTFLAGS=$env:RUSTFLAGS"
Write-Host "   CARGO_TARGET_DIR=$env:CARGO_TARGET_DIR"

# `-Zbuild-std` compiles std out of rust-src.
$installed = @(rustup component list --toolchain $env:RUSTUP_TOOLCHAIN --installed)
if (-not ($installed | Where-Object { $_.Trim() -eq 'rust-src' })) {
    Write-Host "installing rust-src for $env:RUSTUP_TOOLCHAIN (-Zbuild-std needs std's sources)"
    rustup component add rust-src --toolchain $env:RUSTUP_TOOLCHAIN
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host '== trunk build --release'
trunk build --release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host '== stage-packs'
& (Join-Path $root 'scripts\stage-packs.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ---------------------------------------------------------------------------------------------
# Verify the module actually came out threaded, from the build's own output.
# ---------------------------------------------------------------------------------------------
$wasm = Get-ChildItem -Path (Join-Path $root 'web\dist') -Filter 'wenilla_bg*.wasm' -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($null -eq $wasm) {
    [Console]::Error.WriteLine('no wenilla_bg*.wasm in web/dist - the build did not get that far')
    exit 1
}

Write-Host ('== verify: {0} ({1:N0} MB)' -f $wasm.Name, ($wasm.Length / 1MB))
$report = (& (Join-Path $root 'scripts\wasm-memory.ps1') $wasm.FullName) -join "`n"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$report -split "`n" | ForEach-Object { Write-Host "   $_" }

# All three have to hold. Each is a different way to end up single-threaded without an error.
$fail = $false
if (-not $report.Contains('shared=YES')) {
    [Console]::Error.WriteLine('   FAIL: linear memory is not shared - no worker can attach to it.')
    $fail = $true
}
if (-not $report.Contains('IMPORTED memory')) {
    [Console]::Error.WriteLine('   FAIL: linear memory is DEFINED, not imported - every instance would make its own heap')
    [Console]::Error.WriteLine('         and the workers would share nothing with the page.')
    $fail = $true
}
if (-not $report.Contains('+atomics')) {
    [Console]::Error.WriteLine('   FAIL: no +atomics in target_features - RUSTFLAGS did not reach the cargo Trunk spawns.')
    $fail = $true
}
if ($fail) { exit 1 }
Write-Host '   OK: shared, imported, atomics - this module can take worker threads.'

# ---------------------------------------------------------------------------------------------
# BENILLA_KEEP_NAMES=1: re-emit the glue from the cargo artifact without wasm-opt, so stack traces
# name Rust functions. For localhost and a console, never a site.
# ---------------------------------------------------------------------------------------------
if ($env:BENILLA_KEEP_NAMES -eq '1') {
    Write-Host '== BENILLA_KEEP_NAMES=1: re-emitting unoptimized, named glue'
    $artifact = Join-Path $env:CARGO_TARGET_DIR 'wasm32-unknown-unknown\release\wenilla.wasm'
    if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) {
        [Console]::Error.WriteLine("   no cargo artifact at $artifact")
        exit 1
    }
    # The wasm-bindgen Trunk downloaded, whose version MUST match the crate pin (Trunk.toml).
    $localAppData = Default-Env 'LOCALAPPDATA' (Join-Path $HOME 'AppData\Local')
    $cands = @(
        (Join-Path $localAppData 'trunkrs\trunk\cache\wasm-bindgen-0.2.126\wasm-bindgen.exe'),
        (Join-Path $HOME '.cache\trunk\wasm-bindgen-0.2.126\wasm-bindgen')
    )
    $onPath = Get-Command wasm-bindgen -ErrorAction SilentlyContinue
    if ($onPath) { $cands += $onPath.Source }
    $wbg = $cands | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } | Select-Object -First 1
    if (-not $wbg) {
        [Console]::Error.WriteLine('   no wasm-bindgen 0.2.126 found (Trunk caches one; or cargo install wasm-bindgen-cli)')
        exit 1
    }
    & $wbg --target web --out-dir web/dist --out-name wenilla --no-demangle $artifact
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    # Same rewrite `stage-packs.ps1` does: workers have no import map. Latin-1 keeps every other byte.
    $glue = Join-Path $root 'web\dist\wenilla.js'
    $latin1 = [Text.Encoding]::GetEncoding(28591)
    $text = $latin1.GetString([IO.File]::ReadAllBytes($glue))
    $text = $text.Replace('from "wasi_snapshot_preview1"', 'from "./wasi_stubs.js"')
    [IO.File]::WriteAllBytes($glue, $latin1.GetBytes($text))
    $named = Get-Item -LiteralPath (Join-Path $root 'web\dist\wenilla_bg.wasm')
    Write-Host ('   web/dist now carries the NAMED module ({0:N0} MB); web/site still has the optimized one.' -f ($named.Length / 1MB))
}

Write-Host @'

built. Serve it somewhere cross-origin isolated - the page cannot construct a SharedArrayBuffer
without COOP/COEP, and without one this module does not instantiate at all:

  cargo run --release -p wenilla-host -- --www web/dist --packs web/packs

(`wenilla-host::static_site` and `Trunk.toml`'s `[serve] headers` both set the same pair
`web/_headers` sets on the site, so all three agree.)
'@
