# Re-run every Chakra case of the paper. Usage (from the setep root):
#   powershell -File eval\chakra\run_all.ps1 [-Python python] [-Out eval\chakra]
param([string]$Python = "python", [string]$Out = "eval\chakra")
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
$Sweep = "$PSScriptRoot\chakra_sweep.py"
& $Python $Sweep $Root "$Out\gpt2" "gpt2=models/chatgpt2/chatgpt2.setep" "100000,500,400,300,200,160"
& $Python $Sweep $Root "$Out\switch" "switch=models/switch_base_8/switch_base_8.setep" "100000,4000,2000,1000,400,200,100"
& $Python $Sweep $Root "$Out\llama" "llama=models/llama/llama.setep" "100000000,400000,192000,80000"
& $Python $Sweep $Root "$Out\dsr1" "dsr1=models/deepseek_r1/deepseek_r1.setep" "100000000,1000000,192000"
foreach ($n in 2,4,8,16) {
  & $Python $Sweep $Root "$Out\gpt2" "gpt2=models/chatgpt2/chatgpt2.setep" "600" "replicas=$n"
  & $Python $Sweep $Root "$Out\switch" "switch=models/switch_base_8/switch_base_8.setep" "5200" "replicas=$n"
}
& $Python "$PSScriptRoot\make_table.py" $Out
