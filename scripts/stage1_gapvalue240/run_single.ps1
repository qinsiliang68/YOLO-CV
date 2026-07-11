param(
  [Parameter(Mandatory=$true)][string]$RunNumber,
  [Parameter(Mandatory=$true)][string]$MachineConfig,
  [ValidateSet('prepare','train','evaluate','validate','run')][string]$Action='run',
  [string]$AttemptId=''
)
$N = [int]$RunNumber
$Script = Join-Path $PSScriptRoot ("runs/run_{0:D3}.py" -f $N)
$Args = @($Script, '--machine-config', $MachineConfig, '--action', $Action)
if ($AttemptId) { $Args += @('--attempt-id', $AttemptId) }
python @Args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
