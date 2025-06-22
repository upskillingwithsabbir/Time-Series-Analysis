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
    # Adjusted regex to be more robust
    pattern = re.compile(f"^##\s*{re.escape(class_title.split(':')[0])}.*?\n(.*?)(?=^##\s*Class|\Z)", re.DOTALL | re.MULTILINE | re.IGNORECASE)
    match = pattern.search(all_concepts)
    if match:
        content = match.group(1).strip()
        # Add back the main title for context within the notebook
        return f"# {class_title}\n\n" + content
    return f"# {class_title}\n\nContent not found."

# Function to parse interpretation file into sections
def parse_interpretation(interpretation_text):
    sections = {}
    # Split by H2 headings (##) which seem to denote sections
    raw_sections = re.split(r'(?=^##\s+\d+\.)', interpretation_text, flags=re.MULTILINE)
    for section in raw_sections:
        if not section.strip():
            continue
        title_match = re.match(r'^##\s+(\d+\..*?)$', section, flags=re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
            # Remove the title line from the content
            content = re.sub(r'^##\s+.*$\n?\n?', '', section, count=1, flags=re.MULTILINE).strip()
            sections[title] = content
    return sections

# --- Create Notebook for Class 3 --- 
nb = nbf.v4.new_notebook()

# --- Read Content --- 
concepts_full = read_file("/home/ubuntu/TimeSeriesLectureMaterials/00_Concepts_Outline.md")
class3_code_full = read_file("/home/ubuntu/TimeSeriesLectureMaterials/Class3_ML/Class3_Demo.py")
class3_interpretation_full = read_file("/home/ubuntu/TimeSeriesLectureMaterials/Class3_ML/Class3_Interpretation.md")

# --- Extract Class 3 Concepts --- 
class3_concepts = extract_concepts(concepts_full, "Class 3: Time Series Analysis with ML Approach (Using Prophet, XGBOOST)")

# --- Parse Interpretation --- 
interpretation_sections = parse_interpretation(class3_interpretation_full)

# --- Build Notebook Cells --- 

# Title Cell
nb['cells'].append(nbf.v4.new_markdown_cell("# Class 3: Time Series Analysis with ML Approach"))

# Introduction/Concepts Overview
nb['cells'].append(nbf.v4.new_markdown_cell("## Introduction and Concepts\n\nThis notebook explores Machine Learning approaches for time series forecasting, focusing on Facebook Prophet and XGBoost. We will also discuss appropriate train-test split strategies for time series data and compare the results with traditional statistical methods."))

# Split concepts into sections based on the outline structure (e.g., by topic number)
concept_sections = re.split(r'\n(?=\d+\.\s)', class3_concepts)
# Remove the main title from the list as it's already added
concept_sections = concept_sections[1:] # Skip the title part

# Mapping concepts to code blocks (based on comments in the demo script)
code_blocks = {}
current_block_title = None
current_block_code = []

for line in class3_code_full.split('\n'):
    match = re.match(r'^# --- (\d+)\. (.+) ---', line)
    if match:
        if current_block_title:
            code_blocks[current_block_title] = '\n'.join(current_block_code).strip()
        block_num = int(match.group(1))
        block_name = match.group(2).strip()
        # Try to match block name with concept section start
        found_title = None
        for i, concept in enumerate(concept_sections):
             concept_title_match = re.match(r'^(\d+)\.\s*(.*?)(?=\n\*\*|$)', concept, re.IGNORECASE)
             if concept_title_match and int(concept_title_match.group(1)) == block_num:
                 # Use a simpler title format for matching keys
                 found_title = f"{block_num}. {concept_title_match.group(2).strip()}"
                 break
        # Fallback to code comment title if no concept match
        current_block_title = found_title if found_title else f"{block_num}. {block_name}"
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
for line in class3_code_full.split('\n'):
    if line.startswith("import") or line.startswith("from") or "warnings.filterwarnings" in line:
        imports_code.append(line)
    elif line.strip() and not line.startswith("#"):
        break # Stop after first non-import/comment line
nb['cells'].append(nbf.v4.new_code_cell('\n'.join(imports_code).strip()))

# Interleave Concepts, Code, and Interpretation
processed_concepts = set()

# Define the order based on the code blocks found and concepts
section_order = [
    "1. Loading Data", # Code block 1 maps to general setup
    "1. Different Approach to TRAIN-TEST Split for Time Series Forecasting.", # Concept 1
    "2. Fitting Prophet Model", # Code block 2
    "3. Fitting XGBoost Model", # Code block 3
    "3. Comparing the Results between the Statistical Approach and ML Approach.", # Concept 3
    "4. Comparison (Metrics)" # Code block 4 (ML comparison)
]

# Add initial data loading code first
if "1. Loading Data" in code_blocks:
    nb['cells'].append(nbf.v4.new_markdown_cell("## 1. Data Loading and Preparation"))
    nb['cells'].append(nbf.v4.new_code_cell(code_blocks["1. Loading Data"]))
    nb['cells'].append(nbf.v4.new_markdown_cell("**Output:** The code loads the AAPL dataset, selects the 'Adj Close' price, and splits the data into training and testing sets."))

# Process remaining sections
for section_title in section_order[1:]: 
    is_concept_only = not section_title[0].isdigit() or not section_title[1] == '.' # Check if it's likely a concept title
    block_num_str = section_title.split('.')[0]
    try:
        block_num = int(block_num_str)
    except ValueError:
        block_num = -1 # Assign dummy number for concept-only sections

    # Find and Add Concept Cell
    current_concept = None
    concept_title_to_display = section_title # Default to the section title from order list
    matched_concept_idx = -1

    for idx, concept in enumerate(concept_sections):
        concept_title_match = re.match(r'^(\d+)\.\s*(.*?)(?=\n\*\*|$)', concept, re.IGNORECASE)
        if concept_title_match:
            concept_num = int(concept_title_match.group(1))
            concept_name = concept_title_match.group(2).strip()
            # Match based on number or if the section title contains the concept name
            if (block_num != -1 and concept_num == block_num) or (block_num == -1 and concept_name in section_title):
                 if idx not in processed_concepts:
                    current_concept = concept
                    concept_title_to_display = concept_name
                    processed_concepts.add(idx)
                    matched_concept_idx = idx
                    break
    
    if current_concept:
        concept_text = re.sub(r'^\d+\.\s*(.*?)\n', f'## {block_num if block_num != -1 else matched_concept_idx + 1}. {concept_title_to_display}\n', current_concept, count=1)
        concept_text = concept_text.replace('\n*   **Concept:**', '\n**Concept:**')
        concept_text = concept_text.replace('\n*   **Use Cases:**', '\n**Use Cases:**')
        concept_text = concept_text.replace('\n    *   *', '\n*   *')
        nb['cells'].append(nbf.v4.new_markdown_cell(concept_text.strip()))
    elif is_concept_only: # Add placeholder if concept text wasn't found but expected
        nb['cells'].append(nbf.v4.new_markdown_cell(f"## {block_num if block_num != -1 else ''} {concept_title_to_display}"))
        nb['cells'].append(nbf.v4.new_markdown_cell("*Concept details placeholder* \n (Content not found in outline)"))

    # Add Code Cell (only if it's not a concept-only section)
    code_added = False
    if not is_concept_only and section_title in code_blocks and code_blocks[section_title]:
        code_content = code_blocks[section_title]
        # Remove print statements that duplicate interpretation file content
        cleaned_code = []
        lines = code_content.split('\n')
        for line in lines:
            # Keep essential print statements like summaries or errors
            if line.strip().startswith("print(") and "RMSE:" not in line and "MAE:" not in line and "Saved plot:" not in line and "Error fitting" not in line:
                 continue # Skip simple print outputs, keep metrics/errors
            cleaned_code.append(line)
        nb['cells'].append(nbf.v4.new_code_cell('\n'.join(cleaned_code).strip()))
        code_added = True
    elif not is_concept_only:
        nb['cells'].append(nbf.v4.new_markdown_cell("*Code demonstration placeholder* \n (No specific code block found for this section)"))

    # Add Interpretation Cell (only if code was added or interpretation exists)
    interpretation_key = None
    # Find matching interpretation section
    if section_title.startswith("2."): interpretation_key = "1. Prophet Model"
    elif section_title.startswith("3. Fitting XGBoost"): interpretation_key = "2. XGBoost Model"
    elif section_title.startswith("3. Comparing"): interpretation_key = "3. Comparing Statistical vs. Machine Learning Approaches"
    elif section_title.startswith("4."): interpretation_key = "Comparison Summary" # Placeholder for final metrics print

    interpretation_text = None
    if interpretation_key and interpretation_key in interpretation_sections:
        interpretation_text = interpretation_sections[interpretation_key]
    elif interpretation_key == "Comparison Summary":
        # Extract the final print comparison from the code block 4
        if "4. Comparison (Metrics)" in code_blocks:
            match = re.search(r'print\("\\n--- 4\. ML Model Performance Comparison.*?$\n(.*?)\Z', code_blocks["4. Comparison (Metrics)"], re.MULTILINE | re.DOTALL)
            if match:
                interpretation_text = match.group(1).strip().replace('print(f"','').replace('")', '').replace('{','[').replace('}',']')

    if interpretation_text:
         nb['cells'].append(nbf.v4.new_markdown_cell("**Interpretation:**\n\n" + interpretation_text))
    elif code_added: # Add placeholder if code exists but no interpretation found
         nb['cells'].append(nbf.v4.new_markdown_cell("**Interpretation:**\n\n*Interpretation placeholder* \n (Content not found in interpretation file)"))

# --- Write Notebook File --- 
output_notebook_path = "/home/ubuntu/Class3_ML.ipynb"
with open(output_notebook_path, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook created: {output_notebook_path}")


