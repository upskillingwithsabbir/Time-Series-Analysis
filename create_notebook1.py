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
    pattern = re.compile(f"## {class_title}(.*?)(?=## Class|\Z)", re.DOTALL | re.IGNORECASE)
    match = pattern.search(all_concepts)
    if match:
        # Remove the class title line itself and strip whitespace
        content = match.group(1).strip()
        # Add back the main title for context within the notebook
        return f"# {class_title}\n\n" + content
    return f"# {class_title}\n\nContent not found."

# --- Create Notebook for Class 1 --- 
nb = nbf.v4.new_notebook()

# --- Read Content --- 
concepts_full = read_file("/home/ubuntu/TimeSeriesLectureMaterials/00_Concepts_Outline.md")
class1_code_full = read_file("/home/ubuntu/TimeSeriesLectureMaterials/Class1_Basics/Class1_Demo.py")

# --- Extract Class 1 Concepts --- 
class1_concepts = extract_concepts(concepts_full, "Class 1: Introduction to the Basics of Time Series Analysis")

# --- Build Notebook Cells --- 

# Title Cell
nb['cells'].append(nbf.v4.new_markdown_cell("# Class 1: Introduction to the Basics of Time Series Analysis"))

# Introduction/Concepts Overview
nb['cells'].append(nbf.v4.new_markdown_cell("## Introduction and Concepts\n\nThis notebook covers the fundamental concepts of Time Series Analysis. We will explore data preprocessing, smoothing techniques, decomposition, stationarity, and the tools used to identify model orders."))

# Split concepts into sections based on the outline structure (e.g., by topic number)
concept_sections = re.split(r'\n(?=\d+\.\s)', class1_concepts)
# Add the main title back to the first section if split removed it
if not concept_sections[0].strip().startswith("# Class 1"):
     concept_sections[0] = "# Class 1: Introduction to the Basics of Time Series Analysis\n\n" + concept_sections[0]

# Remove the main title from the list as it's already added
concept_sections = concept_sections[1:] # Skip the title part which is now empty or just the title

# Mapping concepts to code blocks (based on comments in the demo script)
code_blocks = {}
current_block_title = None
current_block_code = []

for line in class1_code_full.split('\n'):
    match = re.match(r'^# --- (\d+)\. (.+) ---', line)
    if match:
        if current_block_title:
            code_blocks[current_block_title] = '\n'.join(current_block_code).strip()
        block_num = int(match.group(1))
        block_name = match.group(2).strip()
        # Try to match block name with concept section start
        # This relies on naming consistency in concepts and code comments
        found_title = None
        for i, concept in enumerate(concept_sections):
             # Match concept titles like "1. Introduction to Time Series Analysis"
             concept_title_match = re.match(r'^(\d+)\.\s*(.*?)(?=\n\*\*|$)', concept, re.IGNORECASE)
             if concept_title_match and int(concept_title_match.group(1)) == block_num:
                 found_title = f"{block_num}. {concept_title_match.group(2).strip()}"
                 break
        current_block_title = found_title if found_title else f"{block_num}. {block_name}" # Fallback to code comment title
        current_block_code = []
    elif current_block_title:
        # Exclude the initial imports and warnings from the first block
        if not (current_block_title.startswith("1.") and ("import" in line or "warnings." in line)):
             current_block_code.append(line)

# Add the last block
if current_block_title:
    code_blocks[current_block_title] = '\n'.join(current_block_code).strip()

# Add imports and setup cell separately at the beginning
imports_code = []
for line in class1_code_full.split('\n'):
    if line.startswith("import") or line.startswith("from") or "warnings.filterwarnings" in line:
        imports_code.append(line)
    elif line.strip() and not line.startswith("#"):
        break # Stop after first non-import/comment line
nb['cells'].append(nbf.v4.new_code_cell('\n'.join(imports_code).strip()))

# Interleave Concepts and Code
concept_index = 0
processed_concepts = set()

for i in range(1, 9): # Iterate through expected section numbers 1 to 8
    # Find the concept section matching the number
    current_concept = None
    current_concept_title_full = None
    for idx, concept in enumerate(concept_sections):
        concept_title_match = re.match(r'^(\d+)\.\s*(.*?)(?=\n\*\*|$)', concept, re.IGNORECASE)
        if concept_title_match and int(concept_title_match.group(1)) == i:
            if idx not in processed_concepts:
                current_concept = concept
                current_concept_title_full = f"{i}. {concept_title_match.group(2).strip()}"
                processed_concepts.add(idx)
                break

    # Add Concept Cell
    if current_concept:
        # Reformat concept text slightly for better markdown rendering
        concept_text = re.sub(r'^\d+\.\s*(.*?)\n', r'## \1\n', current_concept, count=1) # Make title H2
        concept_text = concept_text.replace('\n*   **Concept:**', '\n**Concept:**')
        concept_text = concept_text.replace('\n*   **Use Cases:**', '\n**Use Cases:**')
        concept_text = concept_text.replace('\n    *   *', '\n*   *') # Adjust list indentation
        nb['cells'].append(nbf.v4.new_markdown_cell(concept_text.strip()))
    else:
         # Add a placeholder if concept not found (should not happen ideally)
         nb['cells'].append(nbf.v4.new_markdown_cell(f"## {i}. [Concept Placeholder]"))

    # Find and Add Code Cell
    code_key_to_find = None
    if current_concept_title_full:
         # Try matching the full title derived from the concept section
         if current_concept_title_full in code_blocks:
              code_key_to_find = current_concept_title_full
         else:
              # Fallback: try matching just the number prefix if full title fails
              simple_title = f"{i}."
              for key in code_blocks.keys():
                   if key.startswith(simple_title):
                        code_key_to_find = key
                        break
    
    if code_key_to_find and code_blocks[code_key_to_find]:
        code_content = code_blocks[code_key_to_find]
        # Add interpretation/explanation based on print statements in the code
        interpretation = []
        lines = code_content.split('\n')
        cleaned_code = []
        for line in lines:
            if line.strip().startswith("print("):
                 # Extract print content for interpretation
                 print_match = re.search(r'print\((f?["\"])(.*?)\1\)', line)
                 if print_match:
                      interp_text = print_match.group(2)
                      # Basic cleaning of f-string formatting for readability
                      interp_text = re.sub(r'{.*?}', '[value]', interp_text)
                      interpretation.append(f"> {interp_text}")
                 # Keep print statements in code for now, user might want them
                 cleaned_code.append(line) 
            else:
                 cleaned_code.append(line)
        
        nb['cells'].append(nbf.v4.new_code_cell('\n'.join(cleaned_code).strip()))
        if interpretation:
             nb['cells'].append(nbf.v4.new_markdown_cell("**Interpretation/Output:**\n\n" + '\n'.join(interpretation)))
    else:
         # Add placeholder if code not found
         nb['cells'].append(nbf.v4.new_markdown_cell("*Code demonstration placeholder* \n (No specific code block found for this section)"))

# Add any remaining concepts that weren't matched by number
for idx, concept in enumerate(concept_sections):
     if idx not in processed_concepts:
          concept_text = re.sub(r'^\d+\.\s*(.*?)\n', r'## \1\n', concept, count=1)
          concept_text = concept_text.replace('\n*   **Concept:**', '\n**Concept:**')
          concept_text = concept_text.replace('\n*   **Use Cases:**', '\n**Use Cases:**')
          concept_text = concept_text.replace('\n    *   *', '\n*   *')
          nb['cells'].append(nbf.v4.new_markdown_cell(concept_text.strip()))
          nb['cells'].append(nbf.v4.new_markdown_cell("*Code demonstration placeholder* \n (No specific code block found for this section)"))

# --- Write Notebook File --- 
output_notebook_path = "/home/ubuntu/Class1_Basics.ipynb"
with open(output_notebook_path, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook created: {output_notebook_path}")


