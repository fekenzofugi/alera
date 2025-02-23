#!/bin/bash

./bin/ollama serve &

pid=$!

sleep 5

echo "Pulling tinyllama model"
ollama pull tinyllama
ollama create mytinyllama -f doorman.modelfile
echo "Doorman Custom Model pulled"

wait $pid