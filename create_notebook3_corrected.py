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

# --- Create Notebook for Class 3 --- 
nb = nbf.v4.new_notebook()

# --- Read Content --- 
concepts_full = read_file("/home/ubuntu/TimeSeriesLectureMaterials/00_Concepts_Outline.md")
class3_code_full = read_file("/home/ubuntu/TimeSeriesLectureMaterials/Class3_ML/Class3_Demo.py")
class3_interpretation_full = read_file("/home/ubuntu/TimeSeriesLectureMaterials/Class3_ML/Class3_Interpretation.md")

# --- Extract Class 3 Concepts --- 
class3_concepts_raw = extract_concepts(concepts_full, "Class 3: Time Series Analysis with ML Approach (Using Prophet, XGBOOST)")
concept_sections = re.split(r'\n(?=\d+\.\s)', class3_concepts_raw)
concept_map = {}
for section in concept_sections:
    match = re.match(r'^(\d+)\.\s*(.*?)(?=\n\*\*|$)', section, re.IGNORECASE)
    if match:
        num = int(match.group(1))
        title = match.group(2).strip()
        concept_map[num] = {'title': title, 'text': section.strip()}

# --- Split Code into Blocks --- 
code_blocks = split_code_into_blocks(class3_code_full)

# --- Parse Interpretation --- 
interpretation_sections = parse_interpretation(class3_interpretation_full)

# --- Build Notebook Cells --- 

# Title Cell
nb['cells'].append(nbf.v4.new_markdown_cell("# Class 3: Time Series Analysis with ML Approach"))

# Introduction Overview
nb['cells'].append(nbf.v4.new_markdown_cell("This notebook explores Machine Learning approaches for time series forecasting, focusing on Facebook Prophet and XGBoost. We will also discuss appropriate train-test split strategies for time series data and compare the results with traditional statistical methods."))

# Add Setup/Imports Code Block
if "0. Setup and Imports" in code_blocks:
    nb['cells'].append(nbf.v4.new_markdown_cell("## 0. Setup and Imports"))
    nb['cells'].append(nbf.v4.new_code_cell(code_blocks["0. Setup and Imports"]))
    del code_blocks["0. Setup and Imports"]

# Define the order based on the code blocks found and concepts
section_order_keys = sorted([key for key in code_blocks.keys() if key.startswith(tuple(f"{i}." for i in range(1, 5)))], key=lambda x: int(x.split('.')[0]))

processed_concepts_nums = set()

# Add Concept 1 (Train/Test Split) first as it's conceptual before coding
if 1 in concept_map:
    concept_info = concept_map[1]
    title_prefix = f"1. {concept_info['title']}"
    nb['cells'].append(nbf.v4.new_markdown_cell(format_concept(concept_info['text'], title_prefix)))
    processed_concepts_nums.add(1)
    # Add note that this is applied in the data loading section
    nb['cells'].append(nbf.v4.new_markdown_cell("*Note: The train-test split strategy discussed above is implemented in the Data Loading and Preparation code block below.*"))

# Process code blocks (Data Loading, Prophet, XGBoost, Comparison)
for code_key in section_order_keys:
    block_num = int(code_key.split('.')[0])
    
    # Find and Add Concept Cell(s) related to this code block number
    concepts_to_add = []
    concept_title_display = code_key # Default display title
    
    # Concept 2 (Prophet/XGBoost) maps to code blocks 2 and 3
    concept_num_to_find = 2 
    if concept_num_to_find in concept_map and concept_num_to_find not in processed_concepts_nums:
         # Only add Concept 2 header once, before Prophet block
         if block_num == 2:
              concept_info = concept_map[concept_num_to_find]
              concept_title_display = f"2. {concept_info['title']}" 
              concepts_to_add.append(format_concept(concept_info['text'], concept_title_display))
              processed_concepts_nums.add(concept_num_to_find)
         elif block_num == 3: # For XGBoost block, just use the code key title
              concept_title_display = code_key
    else:
         # Use code key if no matching concept found or already processed
         concept_title_display = code_key

    # Add concept markdown cell (only adds Concept 2 text before Prophet)
    if concepts_to_add:
         nb['cells'].append(nbf.v4.new_markdown_cell("\n\n---\n\n".join(concepts_to_add)))
    else: # Add simple header based on code key for other blocks
         nb['cells'].append(nbf.v4.new_markdown_cell(f"## {concept_title_display}"))

    # Add Code Cell
    code_content = code_blocks[code_key]
    nb['cells'].append(nbf.v4.new_code_cell(code_content))
    
    # Add Interpretation Cell
    interpretation_key = None
    # Find matching interpretation section
    if code_key.startswith("1."): interpretation_key = "Data Loading Info" # Special case
    elif code_key.startswith("2."): interpretation_key = "1. Prophet Model"
    elif code_key.startswith("3."): interpretation_key = "2. XGBoost Model"
    elif code_key.startswith("4."): interpretation_key = "ML Comparison Summary" # Special case
    
    interpretation_text = "*Interpretation placeholder* \n (No specific interpretation found for this section)"
    if interpretation_key == "Data Loading Info":
        interpretation_text = "**Output:** The code loads the AAPL dataset, selects the 'Adj Close' price, and splits the data into training and testing sets according to the time series split strategy."
    elif interpretation_key == "ML Comparison Summary":
         # Extract the final print comparison from the code block 4
        match = re.search(r'print\("\\n--- 4\. ML Model Performance Comparison.*?$\n(.*?)\Z', code_content, re.MULTILINE | re.DOTALL)
        if match:
            interpretation_text = match.group(1).strip().replace('print(f"','').replace('")','').replace('{','[').replace('}',']')
            interpretation_text = "**ML Model Performance Comparison (Test Set):**\n\n" + interpretation_text + "\n\n*Note: Compare these metrics with those from Class 2 (Statistical Models). See final comparison section below.*"
        else:
            interpretation_text = "See final comparison section below for ML model metrics."
    elif interpretation_key and interpretation_key in interpretation_sections:
        interpretation_text = interpretation_sections[interpretation_key]

    nb['cells'].append(nbf.v4.new_markdown_cell("**Interpretation:**\n\n" + interpretation_text))
    # Remove used code block
    del code_blocks[code_key]

# Add Concept 3 (Comparison Stat vs ML) last
if 3 in concept_map and 3 not in processed_concepts_nums:
    concept_info = concept_map[3]
    title_prefix = f"3. {concept_info['title']}"
    nb['cells'].append(nbf.v4.new_markdown_cell(format_concept(concept_info['text'], title_prefix)))
    processed_concepts_nums.add(3)
    # Add corresponding interpretation section if available
    interp_key = "3. Comparing Statistical vs. Machine Learning Approaches"
    if interp_key in interpretation_sections:
         nb['cells'].append(nbf.v4.new_markdown_cell("**Detailed Comparison:**\n\n" + interpretation_sections[interp_key]))

# Add any remaining concepts/code blocks (shouldn't be any ideally)
# ... (similar logic as in notebook 1/2 if needed)

# --- Write Notebook File --- 
output_notebook_path = "/home/ubuntu/Class3_ML_Corrected.ipynb"
with open(output_notebook_path, 'w') as f:
    nbf.write(nb, f)

print(f"Corrected Notebook created: {output_notebook_path}")

