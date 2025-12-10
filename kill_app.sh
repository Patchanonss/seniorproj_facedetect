#!/bin/bash
echo "🔪 Killing process on Port 8000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Port 8000 is now free."
else
    echo "👍 Port 8000 was already free."
fi
