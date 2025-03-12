# .\recursive_training.ps1 -Rounds 3 -StartingModel "llama3.2" -QAOutputPrefix "test_output_" -TrainOutputPrefix "Test_" -HfToken "your_hf_token_here"

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

    # Define output directories based on the current round and user-specified prefixes
    $qaOutputDir = "$QAOutputPrefix" + "rev$i"
    $trainOutputDir = "$TrainOutputPrefix" + "rev$i"

    # Step 1: Generate QA Data using the current model.
    $cmdGenerate = "./generate_qa_data.ps1 -Threads 8 -qa_outputDir `"$qaOutputDir`" -Model `"$currentModel`""
    Run-CommandWithRetry -Command $cmdGenerate

    # Step 2: Convert QA Output.
    $cmdConvert = ".\convert_qa_output.ps1 -qa_outputDir `"$qaOutputDir`""
    Run-CommandWithRetry -Command $cmdConvert

    # Step 3: Train the model.
    $cmdTrain = "./train_model_torchtune.ps1 -OutputDir `"$trainOutputDir`" -Quantization `"Q4_K_M`" -TrainData `"data.json`" -HfToken `"$HfToken`" -LearningRate 2e-4 -Epochs 3 -BaseModel `"Meta-llama/Llama-3.1-8B-Instruct`" -MaxSeqLength 2048 -BatchSize 1"
    Run-CommandWithRetry -Command $cmdTrain

    # Step 4: Install the model.
    $cmdInstall = ".\install_model.ps1 `"$trainOutputDir`" -Tool `"torchtune`" -OutputDir `"$trainOutputDir`""
    Run-CommandWithRetry -Command $cmdInstall

    # Update the current model for the next round.
    $currentModel = $trainOutputDir

    Write-Output "=== Training round $i completed ===`n"
}

Write-Output "All training rounds completed successfully."
