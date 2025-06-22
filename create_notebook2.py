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

# --- Create Notebook for Class 2 --- 
nb = nbf.v4.new_notebook()

# --- Read Content --- 
concepts_full = read_file("/home/ubuntu/TimeSeriesLectureMaterials/00_Concepts_Outline.md")
class2_code_full = read_file("/home/ubuntu/TimeSeriesLectureMaterials/Class2_Statistical/Class2_Demo.py")
class2_interpretation_full = read_file("/home/ubuntu/TimeSeriesLectureMaterials/Class2_Statistical/Class2_Interpretation.md")

# --- Extract Class 2 Concepts --- 
class2_concepts = extract_concepts(concepts_full, "Class 2: Time Series Analysis with Statistical Modeling")

# --- Parse Interpretation --- 
interpretation_sections = parse_interpretation(class2_interpretation_full)

# --- Build Notebook Cells --- 

# Title Cell
nb['cells'].append(nbf.v4.new_markdown_cell("# Class 2: Time Series Analysis with Statistical Modeling"))

# Introduction/Concepts Overview
nb['cells'].append(nbf.v4.new_markdown_cell("## Introduction and Concepts\n\nThis notebook delves into statistical models commonly used for time series analysis and forecasting. We will cover AR, MA, ARMA, ARIMA, SARIMAX, and GARCH models, along with model selection criteria and diagnostic checks."))

# Split concepts into sections based on the outline structure (e.g., by topic number)
concept_sections = re.split(r'\n(?=\d+\.\s)', class2_concepts)
# Remove the main title from the list as it's already added
concept_sections = concept_sections[1:] # Skip the title part

# Mapping concepts to code blocks (based on comments in the demo script)
code_blocks = {}
current_block_title = None
current_block_code = []

for line in class2_code_full.split('\n'):
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
for line in class2_code_full.split('\n'):
    if line.startswith("import") or line.startswith("from") or "warnings.filterwarnings" in line:
        imports_code.append(line)
    elif line.strip() and not line.startswith("#"):
        break # Stop after first non-import/comment line
nb['cells'].append(nbf.v4.new_code_cell('\n'.join(imports_code).strip()))

# Interleave Concepts, Code, and Interpretation
concept_index = 0
processed_concepts = set()

# Define the order based on the code blocks found
section_order = [
    "1. Loading Data", # Code block 1 maps to general setup
    "2. Fitting ARIMA Model",
    "3. Fitting AUTO ARIMA Model",
    "4. Fitting SARIMAX Model (with Exogenous Variable)",
    "5. Fitting GARCH Model on Log Returns",
    "6. Diagnostic Checks (Example: Auto ARIMA)",
    "7. Model Performance Comparison (Test Set)"
]

# Add initial data loading code first
if "1. Loading Data" in code_blocks:
    nb['cells'].append(nbf.v4.new_markdown_cell("## 1. Data Loading and Preparation"))
    nb['cells'].append(nbf.v4.new_code_cell(code_blocks["1. Loading Data"]))
    nb['cells'].append(nbf.v4.new_markdown_cell("**Output:** The code loads the AAPL dataset, selects the 'Adj Close' price, creates a log returns series for GARCH, and splits the data into training and testing sets."))

# Process remaining sections
for section_title in section_order[1:]: # Start from section 2
    block_num = int(section_title.split('.')[0])
    
    # Find and Add Concept Cell
    current_concept = None
    concept_title_to_display = f"Concept Placeholder for Section {block_num}"
    for idx, concept in enumerate(concept_sections):
        concept_title_match = re.match(r'^(\d+)\.\s*(.*?)(?=\n\*\*|$)', concept, re.IGNORECASE)
        if concept_title_match and int(concept_title_match.group(1)) == block_num:
             if idx not in processed_concepts:
                current_concept = concept
                concept_title_to_display = concept_title_match.group(2).strip()
                processed_concepts.add(idx)
                break
        # Handle combined concepts like AR/MA/ARMA (maps to block 2)
        elif block_num == 2 and concept_title_match and concept_title_match.group(2).strip().startswith("AR (Autoregressive)"):
             if idx not in processed_concepts:
                current_concept = concept
                concept_title_to_display = "AR, MA, ARMA Models"
                processed_concepts.add(idx)
                # Potentially mark adjacent MA, ARMA concepts as processed too if needed
                break
        # Map GARCH concept (maps to block 5)
        elif block_num == 5 and concept_title_match and concept_title_match.group(2).strip().startswith("ARCH (Autoregressive Conditional Heteroskedasticity)"):
             if idx not in processed_concepts:
                current_concept = concept
                concept_title_to_display = "ARCH/GARCH Models"
                processed_concepts.add(idx)
                break
        # Map Diagnostics concept (maps to block 6)
        elif block_num == 6 and concept_title_match and concept_title_match.group(2).strip().startswith("Diagnostic Check of the Model"):
             if idx not in processed_concepts:
                current_concept = concept
                concept_title_to_display = "Model Diagnostics"
                processed_concepts.add(idx)
                break
        # Map Comparison concept (maps to block 7)
        elif block_num == 7 and concept_title_match and concept_title_match.group(2).strip().startswith("Comparing the Model Performances"):
             if idx not in processed_concepts:
                current_concept = concept
                concept_title_to_display = "Model Performance Comparison"
                processed_concepts.add(idx)
                break

    if current_concept:
        concept_text = re.sub(r'^\d+\.\s*(.*?)\n', f'## {block_num}. {concept_title_to_display}\n', current_concept, count=1)
        concept_text = concept_text.replace('\n*   **Concept:**', '\n**Concept:**')
        concept_text = concept_text.replace('\n*   **Use Cases:**', '\n**Use Cases:**')
        concept_text = concept_text.replace('\n    *   *', '\n*   *')
        nb['cells'].append(nbf.v4.new_markdown_cell(concept_text.strip()))
    else:
        nb['cells'].append(nbf.v4.new_markdown_cell(f"## {block_num}. {concept_title_to_display}"))

    # Add Code Cell
    if section_title in code_blocks and code_blocks[section_title]:
        code_content = code_blocks[section_title]
        # Remove print statements that duplicate interpretation file content
        cleaned_code = []
        lines = code_content.split('\n')
        for line in lines:
            # Keep essential print statements like summaries
            if line.strip().startswith("print(") and "summary()" not in line and "RMSE:" not in line and "MAE:" not in line and "Saved plot:" not in line:
                 continue # Skip simple print outputs, keep summaries/metrics
            cleaned_code.append(line)
        nb['cells'].append(nbf.v4.new_code_cell('\n'.join(cleaned_code).strip()))
    else:
        nb['cells'].append(nbf.v4.new_markdown_cell("*Code demonstration placeholder* \n (No specific code block found for this section)"))

    # Add Interpretation Cell
    interpretation_key = None
    # Find matching interpretation section (adjust title matching as needed)
    if section_title.startswith("2."): interpretation_key = "2. Model Summaries Interpretation" # ARIMA is part of this
    elif section_title.startswith("3."): interpretation_key = "2. Model Summaries Interpretation" # Auto ARIMA is part of this
    elif section_title.startswith("4."): interpretation_key = "2. Model Summaries Interpretation" # SARIMAX is part of this
    elif section_title.startswith("5."): interpretation_key = "2. Model Summaries Interpretation" # GARCH is part of this
    elif section_title.startswith("6."): interpretation_key = "3. Diagnostic Checks (Auto ARIMA Residuals)"
    elif section_title.startswith("7."): interpretation_key = "1. Model Performance Comparison (Test Set Forecasting)"
    
    # Extract specific part for the current model if possible (e.g., from the combined summary section)
    interpretation_text = "Interpretation placeholder."
    if interpretation_key and interpretation_key in interpretation_sections:
        full_interp_section = interpretation_sections[interpretation_key]
        # Simple extraction logic (can be refined)
        if section_title.startswith("2."): # ARIMA
            match = re.search(r'^\*\s*\*ARIMA\(1,1,1\) Summary:\*\*.*?$(.*?)(?=^\*\s*\*|\Z)', full_interp_section, re.MULTILINE | re.DOTALL)
            interpretation_text = match.group(1).strip() if match else "See Model Summaries Interpretation section."
        elif section_title.startswith("3."): # Auto ARIMA
            match = re.search(r'^\*\s*\*Auto ARIMA Summary:\*\*.*?$(.*?)(?=^\*\s*\*|\Z)', full_interp_section, re.MULTILINE | re.DOTALL)
            interpretation_text = match.group(1).strip() if match else "See Model Summaries Interpretation section."
            # Add note about failure
            interpretation_text += "\n\n*Note: Auto ARIMA failed during the prediction phase in the demonstration code.*"
        elif section_title.startswith("4."): # SARIMAX
            match = re.search(r'^\*\s*\*SARIMAX\(1,1,1\) Summary:\*\*.*?$(.*?)(?=^\*\s*\*|\Z)', full_interp_section, re.MULTILINE | re.DOTALL)
            interpretation_text = match.group(1).strip() if match else "See Model Summaries Interpretation section."
        elif section_title.startswith("5."): # GARCH
            match = re.search(r'^\*\s*\*GARCH\(1,1\) Summary.*?$(.*?)(?=^##|\Z)', full_interp_section, re.MULTILINE | re.DOTALL)
            interpretation_text = match.group(1).strip() if match else "See Model Summaries Interpretation section."
        elif section_title.startswith("6."): # Diagnostics
             interpretation_text = full_interp_section
        elif section_title.startswith("7."): # Comparison
             interpretation_text = full_interp_section
        else:
             interpretation_text = full_interp_section # Default to full section if specific part not found

    nb['cells'].append(nbf.v4.new_markdown_cell("**Interpretation:**\n\n" + interpretation_text))

# Add final overall interpretation/recommendations from the file
if "4. Overall Findings & Recommendations for Lecture" in interpretation_sections:
     nb['cells'].append(nbf.v4.new_markdown_cell("## Overall Findings & Recommendations"))
     nb['cells'].append(nbf.v4.new_markdown_cell(interpretation_sections["4. Overall Findings & Recommendations for Lecture"]))

# --- Write Notebook File --- 
output_notebook_path = "/home/ubuntu/Class2_Statistical.ipynb"
with open(output_notebook_path, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook created: {output_notebook_path}")


