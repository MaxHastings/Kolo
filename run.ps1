# Stop script on error
$ErrorActionPreference = "Stop"

# Run the container
Write-Host "Running Docker container..."
docker run --gpus all -p 2222:22 -p 8080:8080 -it -d --name kolo_container kolo