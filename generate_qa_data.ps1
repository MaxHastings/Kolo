<#
.SYNOPSIS
    Runs the generate_qa_data.py script inside the kolo_container Docker container.
.DESCRIPTION
    This script optionally accepts an OpenAI API key as well as worker parameters, a model string, and an output directory,
    sets the API key as an environment variable inside the container, and passes the worker, model, and output directory parameters
    to the Python script.
.EXAMPLE
    .\generate_qa_data.ps1 -OpenAI_API_KEY "your_api_key_here" -Threads 8 -Model "your_model" -qa_outputDir "/path/to/output"
    .\generate_qa_data.ps1 -Threads 8 -Model "your_model" -qa_outputDir "/path/to/output"
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false, HelpMessage = "Provide your OpenAI API Key (optional).")]
    [string]$OpenAI_API_KEY,
    
    [Parameter(Mandatory = $false, HelpMessage = "Max workers for processing.")]
    [int]$Threads = 8,

    [Parameter(Mandatory = $false, HelpMessage = "Directory to output QA results.")]
    [string]$qa_outputDir,

    [Parameter(Mandatory = $false, HelpMessage = "Model string to pass to the Python script.")]
    [string]$Model
)

# Define the container name
$ContainerName = "kolo_container"

# Check if the container is running
$containerRunning = docker ps --format "{{.Names}}" | Select-String -Pattern "$ContainerName"

if (-Not $containerRunning) {
    Write-Host "Error: Container '$ContainerName' is not running." -ForegroundColor Red
    exit 1
}

# Build the base command string to execute inside the container.
$baseCommand = "source /opt/conda/bin/activate kolo_env && python /app/generate_qa_data.py --threads $Threads"

# Append the model argument if provided
if ($Model) {
    $baseCommand += " --model $Model"
}

# Append the qa_outputDir argument if provided
if ($qa_outputDir) {
    $baseCommand += " --qa_output $qa_outputDir"
}

if ($OpenAI_API_KEY) {
    $command = "export OPENAI_API_KEY='$OpenAI_API_KEY'; $baseCommand"
}
else {
    $command = $baseCommand
}

# Execute the Python script inside the container
try {
    Write-Host "Executing generate_qa_data.py inside container: $ContainerName..." -ForegroundColor Cyan
    docker exec -it $ContainerName /bin/bash -c $command

    if ($?) {
        Write-Host "QA data generation script executed successfully!" -ForegroundColor Green
    }
    else {
        Write-Host "Failed to execute the QA data generation script." -ForegroundColor Red
    }
}
catch {
    Write-Host "An error occurred: $_" -ForegroundColor Red
}
