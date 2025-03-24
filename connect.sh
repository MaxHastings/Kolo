#!/bin/bash

# Default to not clearing
CLEAR=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
   case $1 in
       -c|--clear)
           CLEAR=true
           shift
           ;;
       *)
           shift
           ;;
   esac
done

if $CLEAR; then
   echo "Clearing SSH keychain..."
   ssh-keygen -R "[localhost]:2222"
fi

echo "Connecting to remote server..."
ssh root@localhost -p 2222 -t "cd /app && bash"