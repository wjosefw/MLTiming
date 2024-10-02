#!/bin/bash

# Outer loop from 1 to 8
for j in {3..9..2}; do
    clear  
    echo "Running with outer loop parameter: $i"
    for i in {1..8}; do
        echo "  Running with inner loop parameter: $j"
        
        # Call the Python script with parameters i and j
        python3 MLTiming.py $i $j
    done
done
