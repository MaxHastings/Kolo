param (
    [Parameter(Mandatory = $true)]
    [int]$Rounds,

    [Parameter(Mandatory = $true)]
    [string]$StartingModel,

    [Parameter(Mandatory = $true)]
    [string]$QAOutputPrefix,

    [Parameter(Mandatory = $true)]
    [string]$TrainOutputPrefix,

    [Parameter(Mandatory = $true)]
    [string]$HfToken
)

# Base path for training outputs
$basePath = "/var/kolo_data/torchtune"

function Run-CommandWithRetry {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Command
    )
    
    while ($true) {
        Write-Output "Running: $Command"
        Invoke-Expression $Command
        if ($LASTEXITCODE -eq 0) {
            Write-Output "Command completed successfully.`n"
            break
        }
        else {
            Write-Output "Command failed with exit code $LASTEXITCODE. Retrying in 30 seconds..."
            Start-Sleep -Seconds 30
        }
    }
}

# Set the initial model to the user-provided value
$currentModel = $StartingModel

for ($i = 1; $i -le $Rounds; $i++) {
    Write-Output "=== Starting training round $i ==="

    # Define output directories based on the current round and user-specified prefixes.
    $qaOutputDir = "$QAOutputPrefix" + "rev$i"
    # Build the relative train output directory and then the full path.
    $trainOutputDir = "$TrainOutputPrefix" + "rev$i"
    $fullTrainOutputDir = "$basePath/$trainOutputDir"
    
    # --- Check if the fine-tuned model is already installed ---
    $cmdList = "docker exec kolo_container ollama list"
    Write-Output "Checking installed models in Ollama..."
    $installedModelsOutput = Invoke-Expression $cmdList
    Write-Output $installedModelsOutput

    if ($installedModelsOutput -match $trainOutputDir) {
        Write-Output "Model '$trainOutputDir' is already installed. Skipping round $i."
        continue
    }

    # --- Check inside the docker container for the model file ---
    $dockerCheckCmd = "docker exec kolo_container bash -c 'if [ -f ""$fullTrainOutputDir/ModelfileQ4_K_M"" ]; then echo exists; fi'"
    $fileCheckOutput = Invoke-Expression $dockerCheckCmd
    if ($fileCheckOutput -match "exists") {
        Write-Output "Model file found inside docker container at '$fullTrainOutputDir/ModelfileQ4_K_M'. Skipping training steps and proceeding to install model."
        $skipTraining = $true
    }
    else {
        $skipTraining = $false
    }

    if (-not $skipTraining) {
        # Step 1: Generate QA Data using the current model.
        $cmdGenerate = "./generate_qa_data.ps1 -Threads 1 -qa_outputDir `"$qaOutputDir`" -Model `"$currentModel`""
        Run-CommandWithRetry -Command $cmdGenerate

        # Step 2: Convert QA Output.
        $cmdConvert = ".\convert_qa_output.ps1 -qa_outputDir `"$qaOutputDir`""
        Run-CommandWithRetry -Command $cmdConvert

        # Step 3: Train the model.
        # Check if there is any epoch folder (e.g., epoch_1, epoch_2, etc.) in the train output directory inside the docker container.
        $useCheckpoint = ""
        Write-Output "Checking for epoch folder inside docker container in $fullTrainOutputDir"
        $dockerCmd = "docker exec kolo_container bash -c 'ls -d $fullTrainOutputDir/epoch_* 2>/dev/null'"
        $epochFolders = Invoke-Expression $dockerCmd
        if ($epochFolders -and $epochFolders.Trim() -ne "") {
            $useCheckpoint = " -UseCheckpoint"
        }
        
        $cmdTrain = "./train_model_torchtune.ps1 -OutputDir `"$trainOutputDir`" -Quantization `"Q4_K_M`" -TrainData `"data.json`" -HfToken `"$HfToken`" -LearningRate 2e-4 -Epochs 3 -BaseModel `"Meta-llama/Llama-3.1-8B-Instruct`" -MaxSeqLength 2048 -BatchSize 1$useCheckpoint"
        Run-CommandWithRetry -Command $cmdTrain
    }

    # Step 4: Install the model.
    $cmdInstall = ".\install_model.ps1 `"$trainOutputDir`" -Tool `"torchtune`" -OutputDir `"$trainOutputDir`" -Quantization `"Q4_K_M`""
    Run-CommandWithRetry -Command $cmdInstall

    # Update the current model for the next round.
    $currentModel = $trainOutputDir

    Write-Output "=== Training round $i completed ===`n"
}

Write-Output "All training rounds completed successfully."
