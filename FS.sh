#!/bin/bash

# Outer loop from 1 to 8
for j in {43..51..1}; do
    clear  
    echo "Running with outer loop parameter: $i"
    for i in {72..77..1}; do
        echo "  Running with inner loop parameter: $j"
        
        # Call the Python script with parameters i and j
        python3 MLTiming_Convolutional.py $j $i
    done
done
