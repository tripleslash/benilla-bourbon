#Requires -Version 5.1
#
# probe-identity.ps1 - who a scripted run logs in as; dot-sourced by smoke.ps1, cine.ps1 and summon-live.ps1, never run alone.
# The PowerShell twin of scripts/probe-identity.sh; that header holds the why, and the twins change together.
#
# `probe_identity <tag> <tree root>` sets $PROBE_USER, $PROBE_PASS, $PROBE_CHAR, $PROBE_DECLARED
# (non-empty when the file supplied them), $PROBE_HOST and $PROBE_AUTH_PORT in the caller's script
# scope, prints the identity it chose and returns $true; a refusal prints why and returns $false.

function probe_identity([string]$tag, [string]$root) {
    $file = [IO.Path]::Combine($root, '.probe-identity')
    $dash = [string][char]0x2014
    $script:PROBE_DECLARED = ''
    if ([IO.File]::Exists($file)) {
        # sed -n 's/^KEY=//p' | head -1: the rest of the first line that starts KEY=.
        $lines = (New-Object Text.UTF8Encoding $false).GetString([IO.File]::ReadAllBytes($file)).Split([char]10)
        $found = @{}
        foreach ($key in @('WOW_USER', 'WOW_PASS', 'WOW_CHAR')) {
            $found[$key] = ''
            foreach ($line in $lines) {
                if ($line.StartsWith($key + '=', [StringComparison]::Ordinal)) {
                    $found[$key] = $line.Substring($key.Length + 1)
                    break
                }
            }
        }
        $script:PROBE_USER = $found['WOW_USER']
        $script:PROBE_PASS = $found['WOW_PASS']
        $script:PROBE_CHAR = $found['WOW_CHAR']
        if (-not $script:PROBE_USER -or -not $script:PROBE_PASS -or -not $script:PROBE_CHAR) {
            Write-Host "${tag}: REFUSING $dash $file is missing one of WOW_USER, WOW_PASS, WOW_CHAR"
            return $false
        }
        $script:PROBE_DECLARED = '1'
        $ignored = ''
        foreach ($v in @('WOW_USER', 'WOW_PASS', 'WOW_CHAR')) {
            if ([Environment]::GetEnvironmentVariable($v)) { $ignored += " $v" }
        }
        if ($ignored) {
            Write-Host "${tag}: ignoring$ignored $dash this checkout declares its own account (.probe-identity)"
        }
        Write-Host "${tag}: $($script:PROBE_USER) / $($script:PROBE_CHAR) (declared by .probe-identity)"
    } else {
        if (-not $env:WOW_USER -or -not $env:WOW_PASS -or -not $env:WOW_CHAR) {
            Write-Host "${tag}: REFUSING $dash no account to log in as. A scripted run takes WOW_USER, WOW_PASS"
            Write-Host '       and WOW_CHAR from the environment, all three, or from a .probe-identity file'
            Write-Host '       at the tree root, naming a test account on your server whose login kicks'
            Write-Host '       nobody (docs/CONTRIBUTING.md, "Setting up").'
            return $false
        }
        $script:PROBE_USER = $env:WOW_USER
        $script:PROBE_PASS = $env:WOW_PASS
        $script:PROBE_CHAR = $env:WOW_CHAR
        Write-Host "${tag}: $($script:PROBE_USER) / $($script:PROBE_CHAR) (from the environment)"
    }
    # The server: WOW_HOST is `host` or `host:port`, the client's own spelling (realmlist.rs).
    $hostSpec = $env:WOW_HOST
    if (-not $hostSpec) { $hostSpec = 'localhost' }
    $first = $hostSpec.IndexOf(':')
    $last = $hostSpec.LastIndexOf(':')
    if ($first -ge 0) { $script:PROBE_HOST = $hostSpec.Substring(0, $first) } else { $script:PROBE_HOST = $hostSpec }
    if ($last -ge 0) { $script:PROBE_AUTH_PORT = $hostSpec.Substring($last + 1) } else { $script:PROBE_AUTH_PORT = '3724' }
    return $true
}

# `probe_server_or_skip <tag>` prints a skip and returns $false when the auth port does not answer,
# or, for a local server, the stock world port 8085.
function probe_server_or_skip([string]$tag) {
    $ports = @($script:PROBE_AUTH_PORT)
    if ($script:PROBE_HOST -eq 'localhost' -or $script:PROBE_HOST -eq '127.0.0.1') { $ports += '8085' }
    foreach ($port in $ports) {
        $open = $false
        $client = New-Object Net.Sockets.TcpClient
        try {
            $client.Connect($script:PROBE_HOST, [int]$port)
            $open = $true
        } catch {
            $open = $false
        } finally {
            $client.Close()
        }
        if (-not $open) {
            Write-Host "${tag}: SKIPPED $([char]0x2014) nothing listening on $($script:PROBE_HOST):$port (auth $($script:PROBE_AUTH_PORT); a local"
            Write-Host '       vmangos also serves the world on 8085). Start your 1.12.1 server, or point'
            Write-Host '       WOW_HOST at it.'
            return $false
        }
    }
    return $true
}
