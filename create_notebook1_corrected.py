import nbformat as nbf
import re

# Function to read content from a file
def read_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: File not found - {filepath}")
        return ""

# Function to extract concepts for a specific class
def extract_concepts(all_concepts, class_title):
    pattern = re.compile(f"^##\s*{re.escape(class_title.split(':')[0])}.*?\n(.*?)(?=^##\s*Class|\Z)", re.DOTALL | re.MULTILINE | re.IGNORECASE)
    match = pattern.search(all_concepts)
    if match:
        content = match.group(1).strip()
        return f"# {class_title}\n\n" + content
    return f"# {class_title}\n\nContent not found."

# Function to split code into blocks based on comments
def split_code_into_blocks(code_full):
    code_blocks = {}
    current_block_title = None
    current_block_code = []
    # Use a generic initial title for imports/setup before the first marker
    initial_setup_title = "0. Setup and Imports"
    code_blocks[initial_setup_title] = [] 

    lines = code_full.split('\n')
    first_marker_found = False

    for line in lines:
        match = re.match(r'^# --- (\d+)\. (.+) ---', line)
        if match:
            first_marker_found = True
            # Save previous block
            if current_block_title:
                code_blocks[current_block_title] = '\n'.join(current_block_code).strip()
            
            block_num = int(match.group(1))
            block_name = match.group(2).strip()
            current_block_title = f"{block_num}. {block_name}"
            current_block_code = [] # Start new block
        elif first_marker_found:
            # Add line to the current numbered block
            if current_block_title:
                 current_block_code.append(line)
        else:
            # Add line to the initial setup block
            code_blocks[initial_setup_title].append(line)

    # Save the last block
    if current_block_title:
        code_blocks[current_block_title] = '\n'.join(current_block_code).strip()
    # Finalize setup block
    code_blocks[initial_setup_title] = '\n'.join(code_blocks[initial_setup_title]).strip()
    
    # Remove setup block if empty
    if not code_blocks[initial_setup_title]:
        del code_blocks[initial_setup_title]
        
    return code_blocks

# Function to format concept text
def format_concept(concept_text, title_prefix):
    # Make title H2
    concept_text = re.sub(r'^\d+\.\s*(.*?)\n', f'## {title_prefix}\n', concept_text, count=1)
    concept_text = concept_text.replace('\n*   **Concept:**', '\n**Concept:**')
    concept_text = concept_text.replace('\n*   **Use Cases:**', '\n**Use Cases:**')
    concept_text = concept_text.replace('\n    *   *', '\n*   *') # Adjust list indentation
    return concept_text.strip()

# --- Create Notebook for Class 1 --- 
nb = nbf.v4.new_notebook()

# --- Read Content --- 
concepts_full = read_file("/home/ubuntu/TimeSeriesLectureMaterials/00_Concepts_Outline.md")
class1_code_full = read_file("/home/ubuntu/TimeSeriesLectureMaterials/Class1_Basics/Class1_Demo.py")

# --- Extract Class 1 Concepts --- 
class1_concepts_raw = extract_concepts(concepts_full, "Class 1: Introduction to the Basics of Time Series Analysis")
concept_sections = re.split(r'\n(?=\d+\.\s)', class1_concepts_raw)
concept_map = {}
for section in concept_sections:
    match = re.match(r'^(\d+)\.\s*(.*?)(?=\n\*\*|$)', section, re.IGNORECASE)
    if match:
        num = int(match.group(1))
        title = match.group(2).strip()
        concept_map[num] = {'title': title, 'text': section.strip()}

# --- Split Code into Blocks --- 
code_blocks = split_code_into_blocks(class1_code_full)

# --- Build Notebook Cells --- 

# Title Cell
nb['cells'].append(nbf.v4.new_markdown_cell("# Class 1: Introduction to the Basics of Time Series Analysis"))

# Introduction Overview
nb['cells'].append(nbf.v4.new_markdown_cell("This notebook covers the fundamental concepts of Time Series Analysis. We will explore data preprocessing, smoothing techniques, decomposition, stationarity, and the tools used to identify model orders using Python."))

# Add Setup/Imports Code Block
if "0. Setup and Imports" in code_blocks:
    nb['cells'].append(nbf.v4.new_markdown_cell("## 0. Setup and Imports"))
    nb['cells'].append(nbf.v4.new_code_cell(code_blocks["0. Setup and Imports"]))
    # Remove from dict so it's not processed again
    del code_blocks["0. Setup and Imports"]

# Iterate through concepts and add corresponding code
for i in range(1, 9): # Iterate through expected concept numbers 1 to 8
    if i in concept_map:
        concept_info = concept_map[i]
        title_prefix = f"{i}. {concept_info['title']}"
        # Add Concept Markdown Cell
        nb['cells'].append(nbf.v4.new_markdown_cell(format_concept(concept_info['text'], title_prefix)))
        
        # Find corresponding code block(s)
        code_found = False
        # Try exact match first (e.g., "1. Loading and Preprocessing Data")
        code_key_exact = f"{i}. {concept_info['title']}" 
        # Construct potential key from code comments (more reliable)
        potential_code_key = None
        for key in code_blocks.keys():
             if key.startswith(f"{i}."): 
                  potential_code_key = key
                  break

        if potential_code_key and potential_code_key in code_blocks:
            code_content = code_blocks[potential_code_key]
            nb['cells'].append(nbf.v4.new_code_cell(code_content))
            code_found = True
            # Optionally add interpretation based on print statements
            interpretation = []
            lines = code_content.split('\n')
            for line in lines:
                if line.strip().startswith("print("):
                    print_match = re.search(r'print\((f?["\"])(.*?)\1\)', line)
                    if print_match:
                        interp_text = print_match.group(2)
                        interp_text = re.sub(r'{.*?}', '[value]', interp_text) # Basic f-string cleaning
                        interpretation.append(f"> {interp_text}")
            if interpretation:
                nb['cells'].append(nbf.v4.new_markdown_cell("**Interpretation/Output:**\n\n" + '\n'.join(interpretation)))
            # Remove the used code block to avoid duplication
            del code_blocks[potential_code_key]
        
        # Add placeholder if no code was found for this concept
        if not code_found:
             nb['cells'].append(nbf.v4.new_markdown_cell("*Code demonstration placeholder* \n (No specific code block found in the demo script for this exact concept, it might be covered implicitly or within another section's code.)"))

# Add any remaining code blocks that weren't matched to a concept
if code_blocks:
    nb['cells'].append(nbf.v4.new_markdown_cell("## Additional Code Demonstrations\n\n(The following code blocks were present in the demonstration script but did not directly map to a specific concept number above.)"))
    for title, code in code_blocks.items():
        nb['cells'].append(nbf.v4.new_markdown_cell(f"### {title}"))
        nb['cells'].append(nbf.v4.new_code_cell(code))

# --- Write Notebook File --- 
output_notebook_path = "/home/ubuntu/Class1_Basics_Corrected.ipynb"
with open(output_notebook_path, 'w') as f:
    nbf.write(nb, f)

print(f"Corrected Notebook created: {output_notebook_path}")

