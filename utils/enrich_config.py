#!/usr/bin/env python3

"""
This script processes a configuration file (in JSON format) by adding descriptions and default values
for parameters from a provided parameter description file. Additionally, it checks for missing parameters
from both the current class and a parent class and adds them at the end of the configuration file.

The script performs the following tasks:
1. **Read Parameter Descriptions**: It reads two text files containing parameter descriptions and their default values:
   - One for the child class (current model).
   - One for the parent class (if applicable).

2. **Process the Original Config**: It then reads an existing config file and:
   - Adds comments above parameters already present in the config, explaining their descriptions and default values.
   - Adds missing parameters at the end of the file, with a comment explaining their description and default value.

3. **Handle Missing Parameters**: If any parameters are not present in the original config:
   - They are added as comments at the end of the file, either from the child class description file or from the parent class description file.
   - An empty line is added after each parameter value (including missing parameters).

4. **Save the New Config**: The script saves the modified config with added comments and missing parameters to a new output file.

Usage:
    python generate_config_with_missing_params.py
Arguments:
    - param_desc_file: Path to a text file containing descriptions and default values for the child class parameters.
    - parent_param_desc_file: Path to a text file containing descriptions and default values for the parent class parameters.
    - config_file: Path to the original config file (in JSON format).
    - output_file: Path where the updated config file will be saved.

This script is useful for ensuring that a configuration file contains all relevant parameter descriptions,
including those from parent classes, and to highlight any missing parameters that need to be added.
"""

import re


# Function to read the parameter descriptions from the given text file
def read_param_descriptions(file_path):
    """
    Reads the parameter descriptions from a given file and returns a dictionary of parameters
    with their description and default value.

    Args:
        file_path (str): Path to the parameter descriptions file.

    Returns:
        dict: A dictionary where keys are parameter names and values are tuples of (description, default_value).
    """
    param_descriptions = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                # Extract the argument name, default value, and description
                parts = line.split(" — ")
                if len(parts) == 2:
                    description = parts[1].strip()
                    param_and_default = parts[0].split(" (")
                    if len(param_and_default) == 2:
                        param_name = param_and_default[0].strip()
                        default_value = param_and_default[1].replace(')', '').strip()
                        param_descriptions[param_name] = (description, default_value)
    return param_descriptions


# Function to read the config file as a plain text while preserving comments
def read_config_file_with_comments(file_path):
    """
    Reads the original config file with comments preserved.

    Args:
        file_path (str): Path to the config file.

    Returns:
        str: The content of the config file as a string.
    """
    with open(file_path, 'r') as file:
        content = file.read()
    return content


# Function to add comments with descriptions and default values before the parameters
def add_comments_to_config(original_content, param_descriptions, parent_param_descriptions):
    """
    Adds comments (description and default value) to the config file for each parameter
    based on the provided parameter descriptions. Also adds missing parameters at the end.

    Args:
        original_content (str): Original config file content with comments.
        param_descriptions (dict): Dictionary of child class parameter descriptions and default values.
        parent_param_descriptions (dict): Dictionary of parent class parameter descriptions and default values.

    Returns:
        str: The updated config content with comments added for each parameter and missing parameters at the end.
    """
    lines = original_content.splitlines()
    new_content = []

    # Keep track of parameters already processed
    processed_params = set()

    # Add comments for existing parameters
    for line in lines:
        match = re.match(r'^\s*"([^"]+)":', line)
        if match:
            param_name = match.group(1)
            if param_name in param_descriptions:
                description, default_value = param_descriptions[param_name]
                # Add the comment with the description first, then the default value, before the parameter line
                new_content.append(f'  // {description}')
                new_content.append(f'  // ({default_value})')
                processed_params.add(param_name)
            elif param_name in parent_param_descriptions and param_name not in processed_params:
                description, default_value = parent_param_descriptions[param_name]
                # If it's from the parent class and not yet commented, add the parent comment
                new_content.append(f'  // {description}')
                new_content.append(f'  // ({default_value})')
                processed_params.add(param_name)

        # Add the original line (whether it's a parameter or a comment)
        new_content.append(line)

        # Add an empty line after each parameter value
        if re.match(r'^\s*"[^"]+":', line):  # If it's a parameter line
            new_content.append('')  # Add an empty line after the parameter value

    # Add missing parameters at the end of the config
    new_content.append('// Missing parameters:')
    for param_name, (description, default_value) in param_descriptions.items():
        if param_name not in processed_params:
            # Add the description and default value for missing parameters as comments at the end
            new_content.append(f'  // {description}')
            new_content.append(f'  // (defaults to {default_value})')
            new_content.append(f'  // {param_name}: {default_value}')  # Placeholder for missing param

    # Now handle missing parameters from the parent class
    for param_name, (description, default_value) in parent_param_descriptions.items():
        if param_name not in processed_params:
            # Add the description and default value for missing parameters as comments at the end
            new_content.append(f'  // {description}')
            new_content.append(f'  // (defaults to {default_value})')
            new_content.append(f'  // {param_name}: {default_value}')  # Placeholder for missing param
            new_content.append("")

    return "\n".join(new_content)


# Main function to read files, process data, and save the new config
def main(param_desc_file, parent_param_desc_file, config_file, output_file):
    """
    Main function to process the original config file, add comments and missing parameters,
    and save the updated config file.

    Args:
        param_desc_file (str): Path to the file containing the child class parameter descriptions.
        parent_param_desc_file (str): Path to the file containing the parent class parameter descriptions.
        config_file (str): Path to the original config file with comments.
        output_file (str): Path to save the new config file with added comments.
    """
    # Read parameter descriptions and the original config file with comments
    param_descriptions = read_param_descriptions(param_desc_file)
    parent_param_descriptions = read_param_descriptions(parent_param_desc_file)
    original_content = read_config_file_with_comments(config_file)

    # Generate the new config with comments
    new_config = add_comments_to_config(original_content, param_descriptions, parent_param_descriptions)

    # Save the new config to the output file
    with open(output_file, 'w') as f:
        f.write(new_config)

    print(f"New config file with comments saved to {output_file}")




# Run the script
if __name__ == "__main__":
    param_desc_file = "config-info.txt"  # Path to the original param descriptions file

    # TODO: set this
    parent_param_desc_file = "parent_param_descriptions.txt"  # Path to the parent class param descriptions file
    config_file = "configs/transformer_bart_medium.json"  # Path to the original config file with comments
    output_file = "new_config_with_comments.json"  # Output file path

    main(param_desc_file, parent_param_desc_file, config_file, output_file)
