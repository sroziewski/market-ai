#!/bin/bash

# Check if the correct number of arguments are given
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 input_file.csv output_file.csv"
  exit 1
fi

input_file=$1   # Input CSV file
output_file=$2  # Output CSV file

# Check if the input file exists
if [ ! -f "$input_file" ]; then
  echo "Error: Input file $input_file not found."
  exit 1
fi

# Get the total number of lines in the file
total_lines=$(wc -l < "$input_file")

# Check if the file is empty
if [ "$total_lines" -le 0 ]; then
  echo "Error: Input file $input_file is empty."
  exit 1
fi

# Calculate 80% of the total lines
lines_to_extract=$(awk "BEGIN {printf \"%d\", $total_lines * 0.8}")

# Extract only the first 80% of the lines and write to the output file
head -n "$lines_to_extract" "$input_file" > "$output_file"

# Confirm the operation
if [ $? -eq 0 ]; then
  echo "Extracted first 80% of rows ($lines_to_extract of $total_lines) from $input_file to $output_file."
else
  echo "Error during extraction."
  exit 1
fi
