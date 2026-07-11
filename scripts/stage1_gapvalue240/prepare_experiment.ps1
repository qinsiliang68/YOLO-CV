param([Parameter(Mandatory=$true)][string]$MachineConfig)
python (Join-Path $PSScriptRoot 'prepare_experiment.py') --machine-config $MachineConfig
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
