#!/bin/bash

# Usage: ./extract_1st_and_last_20_percent.sh input_file.csv output_file.csv

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
if [ "$total_lines" -le 1 ]; then
  echo "Error: Input file is empty or only contains a single line."
  exit 1
fi

# Calculate the starting line for the last 20% of rows
lines_to_extract=$(awk "BEGIN {printf \"%d\", $total_lines * 0.2}")
start_line=$((total_lines - lines_to_extract + 1))

# Extract the first row (header) and the last 20% of rows
{ 
  head -n 1 "$input_file";                   # Extract the first row
  tail -n "$lines_to_extract" "$input_file";  # Extract the last 20% of rows
} > "$output_file"

# Confirm successful operation and provide extracted details
if [ $? -eq 0 ]; then
  echo "Extracted the first row (header) and the last $lines_to_extract rows (20% of $total_lines) from $input_file to $output_file."
else
  echo "Error during extraction."
  exit 1
fi
