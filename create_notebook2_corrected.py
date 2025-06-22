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

# Function to parse interpretation file into sections
def parse_interpretation(interpretation_text):
    sections = {}
    # Split by H2 headings (##) which seem to denote sections
    raw_sections = re.split(r'(?=^##\s+\d+\.)', interpretation_text, flags=re.MULTILINE)
    current_title = "General Introduction" # For content before the first numbered heading
    current_content = []
    for section in raw_sections:
        if not section.strip():
            continue
        title_match = re.match(r'^##\s+(\d+\..*?)$', section, flags=re.MULTILINE)
        if title_match:
            # Save previous section
            if current_title:
                 sections[current_title] = '\n'.join(current_content).strip()
            # Start new section
            current_title = title_match.group(1).strip()
            # Remove the title line from the content for the new section
            current_content = re.sub(r'^##\s+.*$\n?\n?', '', section, count=1, flags=re.MULTILINE).strip().split('\n')
        else:
            # Append to current section (likely the intro part)
            current_content.extend(section.strip().split('\n'))
    # Save the last section
    if current_title:
        sections[current_title] = '\n'.join(current_content).strip()
    return sections

# Function to split code into blocks based on comments
def split_code_into_blocks(code_full):
    code_blocks = {}
    current_block_title = None
    current_block_code = []
    initial_setup_title = "0. Setup and Imports"
    code_blocks[initial_setup_title] = [] 
    lines = code_full.split('\n')
    first_marker_found = False
    for line in lines:
        match = re.match(r'^# --- (\d+)\. (.+) ---', line)
        if match:
            first_marker_found = True
            if current_block_title:
                code_blocks[current_block_title] = '\n'.join(current_block_code).strip()
            block_num = int(match.group(1))
            block_name = match.group(2).strip()
            current_block_title = f"{block_num}. {block_name}"
            current_block_code = [] 
        elif first_marker_found:
            if current_block_title:
                 current_block_code.append(line)
        else:
            code_blocks[initial_setup_title].append(line)
    if current_block_title:
        code_blocks[current_block_title] = '\n'.join(current_block_code).strip()
    code_blocks[initial_setup_title] = '\n'.join(code_blocks[initial_setup_title]).strip()
    if not code_blocks[initial_setup_title]:
        del code_blocks[initial_setup_title]
    return code_blocks

# Function to format concept text
def format_concept(concept_text, title_prefix):
    concept_text = re.sub(r'^\d+\.\s*(.*?)\n', f'## {title_prefix}\n', concept_text, count=1)
    concept_text = concept_text.replace('\n*   **Concept:**', '\n**Concept:**')
    concept_text = concept_text.replace('\n*   **Use Cases:**', '\n**Use Cases:**')
    concept_text = concept_text.replace('\n    *   *', '\n*   *')
    return concept_text.strip()

# --- Create Notebook for Class 2 --- 
nb = nbf.v4.new_notebook()

# --- Read Content --- 
concepts_full = read_file("/home/ubuntu/TimeSeriesLectureMaterials/00_Concepts_Outline.md")
class2_code_full = read_file("/home/ubuntu/TimeSeriesLectureMaterials/Class2_Statistical/Class2_Demo.py")
class2_interpretation_full = read_file("/home/ubuntu/TimeSeriesLectureMaterials/Class2_Statistical/Class2_Interpretation.md")

# --- Extract Class 2 Concepts --- 
class2_concepts_raw = extract_concepts(concepts_full, "Class 2: Time Series Analysis with Statistical Modeling")
concept_sections = re.split(r'\n(?=\d+\.\s)', class2_concepts_raw)
concept_map = {}
for section in concept_sections:
    match = re.match(r'^(\d+)\.\s*(.*?)(?=\n\*\*|$)', section, re.IGNORECASE)
    if match:
        num = int(match.group(1))
        title = match.group(2).strip()
        concept_map[num] = {'title': title, 'text': section.strip()}

# --- Split Code into Blocks --- 
code_blocks = split_code_into_blocks(class2_code_full)

# --- Parse Interpretation --- 
interpretation_sections = parse_interpretation(class2_interpretation_full)

# --- Build Notebook Cells --- 

# Title Cell
nb['cells'].append(nbf.v4.new_markdown_cell("# Class 2: Time Series Analysis with Statistical Modeling"))

# Introduction Overview
nb['cells'].append(nbf.v4.new_markdown_cell("This notebook delves into statistical models commonly used for time series analysis and forecasting. We will cover AR, MA, ARMA, ARIMA, SARIMAX, and GARCH models, along with model selection criteria and diagnostic checks."))

# Add Setup/Imports Code Block
if "0. Setup and Imports" in code_blocks:
    nb['cells'].append(nbf.v4.new_markdown_cell("## 0. Setup and Imports"))
    nb['cells'].append(nbf.v4.new_code_cell(code_blocks["0. Setup and Imports"]))
    del code_blocks["0. Setup and Imports"]

# Define the order based on the code blocks found (more reliable)
section_order_keys = sorted([key for key in code_blocks.keys() if key.startswith(tuple(f"{i}." for i in range(1, 11)))], key=lambda x: int(x.split('.')[0]))

processed_concepts_nums = set()

for code_key in section_order_keys:
    block_num = int(code_key.split('.')[0])
    
    # Find and Add Concept Cell(s) related to this code block number
    concepts_to_add = []
    concept_title_display = code_key # Default display title
    
    # Special handling for combined concepts
    if block_num == 2: # ARIMA code block covers AR, MA, ARMA, ARIMA concepts
        concept_nums_to_find = [2, 3]
    elif block_num == 4: # SARIMAX code block covers ARIMAX, SARIMAX concepts
        concept_nums_to_find = [4, 5]
    elif block_num == 5: # GARCH code block covers ARCH, GARCH concepts
        concept_nums_to_find = [7]
    elif block_num == 6: # Diagnostics code block covers Diagnostics, AIC/BIC concepts
        concept_nums_to_find = [8, 9]
    elif block_num == 7: # Comparison code block covers Comparison concept
        concept_nums_to_find = [10]
    else:
        concept_nums_to_find = [block_num]
        
    first_title_found = False
    for concept_num in concept_nums_to_find:
        if concept_num in concept_map and concept_num not in processed_concepts_nums:
            concept_info = concept_map[concept_num]
            if not first_title_found:
                 # Use the first matched concept's title for the section header
                 concept_title_display = f"{block_num}. {concept_info['title']}" 
                 first_title_found = True
            concepts_to_add.append(format_concept(concept_info['text'], f"{concept_num}. {concept_info['title']}"))
            processed_concepts_nums.add(concept_num)

    # Add combined concept markdown cell
    if concepts_to_add:
         nb['cells'].append(nbf.v4.new_markdown_cell(f"## {concept_title_display}\n\n" + "\n\n---\n\n".join(concepts_to_add)))
    else: # Add placeholder if no concept found for this code block
         nb['cells'].append(nbf.v4.new_markdown_cell(f"## {code_key}")) # Use code key as title
         nb['cells'].append(nbf.v4.new_markdown_cell("*Concept placeholder* \n (No specific concept outline found for this section)"))

    # Add Code Cell
    code_content = code_blocks[code_key]
    nb['cells'].append(nbf.v4.new_code_cell(code_content))
    
    # Add Interpretation Cell
    interpretation_key = None
    # Find matching interpretation section (adjust title matching as needed)
    if code_key.startswith("2."): interpretation_key = "2. Model Summaries Interpretation" # ARIMA
    elif code_key.startswith("3."): interpretation_key = "2. Model Summaries Interpretation" # Auto ARIMA
    elif code_key.startswith("4."): interpretation_key = "2. Model Summaries Interpretation" # SARIMAX
    elif code_key.startswith("5."): interpretation_key = "2. Model Summaries Interpretation" # GARCH
    elif code_key.startswith("6."): interpretation_key = "3. Diagnostic Checks (Auto ARIMA Residuals)"
    elif code_key.startswith("7."): interpretation_key = "1. Model Performance Comparison (Test Set Forecasting)"
    
    interpretation_text = "*Interpretation placeholder* \n (No specific interpretation found for this section)"
    if interpretation_key and interpretation_key in interpretation_sections:
        full_interp_section = interpretation_sections[interpretation_key]
        # Simple extraction logic (can be refined)
        if code_key.startswith("2."): # ARIMA
            match = re.search(r'^\*\s*\*ARIMA\(1,1,1\) Summary:\*\*.*?$(.*?)(?=^\*\s*\*|\Z)', full_interp_section, re.MULTILINE | re.DOTALL)
            interpretation_text = match.group(1).strip() if match else full_interp_section
        elif code_key.startswith("3."): # Auto ARIMA
            match = re.search(r'^\*\s*\*Auto ARIMA Summary:\*\*.*?$(.*?)(?=^\*\s*\*|\Z)', full_interp_section, re.MULTILINE | re.DOTALL)
            interpretation_text = match.group(1).strip() if match else full_interp_section
            interpretation_text += "\n\n*Note: Auto ARIMA failed during the prediction phase in the demonstration code.*"
        elif code_key.startswith("4."): # SARIMAX
            match = re.search(r'^\*\s*\*SARIMAX\(1,1,1\) Summary:\*\*.*?$(.*?)(?=^\*\s*\*|\Z)', full_interp_section, re.MULTILINE | re.DOTALL)
            interpretation_text = match.group(1).strip() if match else full_interp_section
        elif code_key.startswith("5."): # GARCH
            match = re.search(r'^\*\s*\*GARCH\(1,1\) Summary.*?$(.*?)(?=^##|\Z)', full_interp_section, re.MULTILINE | re.DOTALL)
            interpretation_text = match.group(1).strip() if match else full_interp_section
        else: # Diagnostics, Comparison
             interpretation_text = full_interp_section

    nb['cells'].append(nbf.v4.new_markdown_cell("**Interpretation:**\n\n" + interpretation_text))
    # Remove used code block
    del code_blocks[code_key]

# Add any remaining concepts that weren't matched
for concept_num in sorted(concept_map.keys()):
    if concept_num not in processed_concepts_nums:
        concept_info = concept_map[concept_num]
        title_prefix = f"{concept_num}. {concept_info['title']}"
        nb['cells'].append(nbf.v4.new_markdown_cell(format_concept(concept_info['text'], title_prefix)))
        nb['cells'].append(nbf.v4.new_markdown_cell("*Code demonstration placeholder* \n (No specific code block found in the demo script for this exact concept.)"))

# Add final overall interpretation/recommendations from the file
if "4. Overall Findings & Recommendations for Lecture" in interpretation_sections:
     nb['cells'].append(nbf.v4.new_markdown_cell("## Overall Findings & Recommendations"))
     nb['cells'].append(nbf.v4.new_markdown_cell(interpretation_sections["4. Overall Findings & Recommendations for Lecture"]))

# --- Write Notebook File --- 
output_notebook_path = "/home/ubuntu/Class2_Statistical_Corrected.ipynb"
with open(output_notebook_path, 'w') as f:
    nbf.write(nb, f)

print(f"Corrected Notebook created: {output_notebook_path}")

