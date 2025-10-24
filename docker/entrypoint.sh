#!/bin/bash
set -e

# Start Ollama in background
echo "🚀 Starting Ollama daemon..."
ollama serve > /var/log/ollama.log 2>&1 &
sleep 3

# Pull models if specified
if [ -n "$OLLAMA_MODELS" ]; then
  echo "📦 Checking Ollama models: $OLLAMA_MODELS"

  for model in $OLLAMA_MODELS; do
    if ollama list | grep -q "^$model"; then
      echo "✅ Model $model already present, skipping download."
    else
      echo "⬇️  Pulling missing model: $model ..."
      if ollama pull "$model"; then
        echo "✅ Successfully pulled $model"
      else
        echo "⚠️ Failed to pull $model"
      fi
    fi
  done
else
  echo "ℹ️ No OLLAMA_MODELS specified to pre-download."
fi

# Start your app
echo "🚀 Starting RAG application..."
exec "$@"