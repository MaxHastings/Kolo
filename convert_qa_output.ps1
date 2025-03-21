# run_parse_and_convert.ps1
#
# This script will:
# 1. Run parse_qa_data.py with:
#       --input_dir /app/qa_generation_output 
#       --output_file /app/data.jsonl
#
# 2. Then call convert_jsonl_to_json.py to convert the JSONL file (/app/data.jsonl)
#    into a JSON file (/app/data.json)
#
# Note: Adjust the file names if your actual use case differs.

# Define fixed values for directories, file names, and container/environment
$inputDir = "/var/kolo_data/qa_generation_output"
$qaJsonlFile = "/app/data.jsonl"  # File generated by parse_qa_data.py (JSONL format)
$finalJsonFile = "/app/data.json"   # File produced by convert_jsonl_to_json.py
$containerName = "kolo_container"
$envActivate = "source /opt/conda/bin/activate kolo_env"

# Step 1: Run parse_qa_data.py inside the container
try {
    Write-Host "Running parse_qa_data.py in container $containerName..."
    docker exec -it $containerName bash -c "$envActivate && python /app/parse_qa_data.py"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "parse_qa_data.py executed successfully." -ForegroundColor Green
    }
    else {
        Write-Host "Failed to run parse_qa_data.py." -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "An error occurred while running parse_qa_data.py: $_" -ForegroundColor Red
    exit 1
}

# Step 2: Run convert_jsonl_to_json.py inside the container
try {
    Write-Host "Running convert_jsonl_to_json.py in container $containerName..."
    docker exec -it $containerName bash -c "$envActivate && python /app/convert_jsonl_to_json.py '$qaJsonlFile' '$finalJsonFile'"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Conversion successful! JSON file created at $finalJsonFile." -ForegroundColor Green
    }
    else {
        Write-Host "Conversion script failed." -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "An error occurred while running convert_jsonl_to_json.py: $_" -ForegroundColor Red
    exit 1
}

