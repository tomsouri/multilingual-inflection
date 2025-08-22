#!/usr/bin/env python3
"""
This script processes a TSV file containing parameter placeholders and their default values,
replaces those placeholders in a given script, and generates new script files with optional values.

It performs the following tasks:
1. Reads the TSV file to extract placeholders and their corresponding values.
2. Replaces placeholders in the base script with default values.
3. Generates additional scripts with optional values, replacing specific placeholders as required.

Usage:
    python script.py params.tsv run-trm-combination-of-best.sh run-trm-ft-base.sh
"""

import sys
import os

# Function to read TSV and create a dictionary of parameters with their values
def read_tsv(tsv_file):
    """Reads a TSV file and extracts parameter placeholders, default values, and optional values.
    
    Args:
        tsv_file (str): Path to the TSV file containing parameters.
    
    Returns:
        tuple: A dictionary mapping placeholders to default values, and a list of optional values.
    """
    param_dict = {}
    optional_values = []
    with open(tsv_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:  # Ensure there is a default value
                param_dict[parts[1]] = parts[2]  # Placeholder -> Default value
                if len(parts) > 3:
                    optional_values.append((parts[0], parts[1], parts[3:]))  # Parameter Name, Placeholder, Optional values
    print(f"Loaded parameters: {param_dict}")
    return param_dict, optional_values

# Function to replace placeholders in the script
def replace_placeholders(script_file, param_dict, output_file):
    """Replaces placeholders in the script with their default values.
    
    Args:
        script_file (str): Path to the input script file.
        param_dict (dict): Dictionary mapping placeholders to default values.
        output_file (str): Path to save the updated script.
    """
    with open(script_file, 'r') as f:
        script_content = f.read()
    
    for placeholder, default_value in param_dict.items():
        if placeholder in script_content:
            print(f"Replacing {placeholder} with {default_value}")
        script_content = script_content.replace(placeholder, default_value)

    script_content.replace("<PAR>=<VAL>", "base")
    
    with open(output_file, 'w') as f:
        f.write(script_content)
    print(f"Updated script saved as {output_file}")

# Function to handle optional values
def process_optional_values(script_file, param_dict, optional_values):
    """Generates additional scripts with optional values, replacing specific placeholders.
    
    Args:
        script_file (str): Path to the input script template.
        param_dict (dict): Dictionary mapping placeholders to default values.
        optional_values (list): List of tuples containing parameter names, placeholders, and optional values.
    """
    with open(script_file, 'r') as f:
        script_template = f.read()
    
    for param_name, placeholder, values in optional_values:
        for value in values:
            script_content = script_template.replace(placeholder, value)
            script_content = script_content.replace("<VAL>", value)
            script_content = script_content.replace("<PAR>", param_name)
            
            for default_placeholder, default_value in param_dict.items():
                script_content = script_content.replace(default_placeholder, default_value)
            
            output_filename = f"run-trm-ft-{param_name}-{value}.sh"
            with open(output_filename, 'w') as f:
                f.write(script_content)
            print(f"Created script: {output_filename}")

# Main function
def main():
    tsv_file = "params.tsv"
    script_file = "run-trm-combination-of-best.sh"
    output_file = "run-trm-ft-base.sh"
    
    param_dict, optional_values = read_tsv(tsv_file)
    replace_placeholders(script_file, param_dict, output_file)
    process_optional_values(script_file, param_dict, optional_values)

if __name__ == "__main__":
    main()
