#!/bin/bash

# File names for logs
VLLM_LOG="vllm_server.log"
REQUEST_LOG="request.log"

# Function to get current timestamp in human-readable format
get_timestamp() {
    date '+%Y-%m-%d %H:%M:%S.%3N'
}

# Start time
START_TIME=$(get_timestamp)

# Start vLLM server in background and redirect output to log file
echo "Starting vLLM server at $(get_timestamp)" | tee -a "$VLLM_LOG"
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-13b-chat-hf \
    --host 0.0.0.0 \
    --port 8000 > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Start send_request.py in background
echo "Starting request at $(get_timestamp)" | tee -a "$REQUEST_LOG"
python send_request.py > "$REQUEST_LOG" 2>&1 &
REQUEST_PID=$!

# Function to check if send_request.py has completed
check_request_completion() {
    if ! kill -0 $REQUEST_PID 2>/dev/null; then
        wait $REQUEST_PID
        return $?
    fi
    return 1
}

# Wait for send_request.py to finish
echo "Waiting for send_request.py to complete..."
while true; do
    if check_request_completion; then
        REQUEST_EXIT_CODE=$?
        break
    fi
    sleep 1
done

REQUEST_END_TIME=$(get_timestamp)

# Check if send_request.py completed successfully
if [ $REQUEST_EXIT_CODE -eq 0 ] && grep -q "Full response:" "$REQUEST_LOG"; then
    echo "send_request.py completed successfully"
    # Extract time to first token from request log
    TIME_TO_FIRST_TOKEN=$(grep "Time to first token:" "$REQUEST_LOG" | awk '{print $5}')
    # Calculate total time (in seconds)
    TOTAL_TIME=$(echo "$(date -d "$REQUEST_END_TIME" +%s.%N) - $(date -d "$START_TIME" +%s.%N)" | bc)
    # Print results
    echo "----------------------------------------"
    echo "Results:"
    echo "Script start time: $START_TIME"
    echo "Time to first token: $TIME_TO_FIRST_TOKEN seconds"
    echo "Total script duration: $TOTAL_TIME seconds"
    echo "----------------------------------------"
    echo "vLLM server log (only [chenw] messages):"
    grep "\[chenw\]" "$VLLM_LOG"
    echo "----------------------------------------"
    echo "Request log (full log):"
    cat "$REQUEST_LOG"
    # Kill the vLLM server only after successful completion
    echo "Stopping vLLM server"
    kill $VLLM_PID
else
    echo "send_request.py did not complete successfully (exit code: $REQUEST_EXIT_CODE)"
    echo "Check $REQUEST_LOG for details"
    echo "vLLM server is still running with PID $VLLM_PID"
fi
