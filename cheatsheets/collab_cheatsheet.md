# Google Colab Cheatsheet for Biology Students

*A comprehensive guide to using Google Colab for biological data analysis and Python programming*

---

## 🚀 Getting Started with Colab

### What is Google Colab?
- **Free cloud-based Jupyter notebook environment**
- **No installation required** - runs in your web browser
- **Pre-installed libraries** for data science and machine learning
- **Free GPU/TPU access** for computational tasks
- **Easy sharing and collaboration** with peers and instructors

### Accessing Colab
1. Go to **colab.research.google.com**
2. Sign in with your Google account
3. Choose: **New notebook** or **Upload** existing notebook
4. Save notebooks to **Google Drive** automatically

---

## 🔧 Essential Colab Interface

### Notebook Structure
| Component | Purpose |
|-----------|---------|
| **Code cells** | Write and execute Python code |
| **Text cells** | Add documentation, notes, equations (Markdown) |
| **Output area** | View results, plots, error messages |
| **Menu bar** | File operations, runtime management |

### Key Shortcuts
| Action | Shortcut |
|--------|----------|
| **Run cell** | `Ctrl + Enter` |
| **Run cell + add new** | `Shift + Enter` |
| **Insert cell above** | `Ctrl + M A` |
| **Insert cell below** | `Ctrl + M B` |
| **Delete cell** | `Ctrl + M D` |
| **Change to code cell** | `Ctrl + M Y` |
| **Change to text cell** | `Ctrl + M M` |
| **Command palette** | `Ctrl + Shift + P` |

---

## 📊 Working with Files and Data

### Mounting Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Access your files
import os
os.listdir('/content/drive/MyDrive')
```

### Uploading Files
```python
from google.colab import files

# Upload from computer
uploaded = files.upload()

# Or drag and drop files into the file browser
```

### Downloading Results
```python
from google.colab import files

# Download a file
files.download('results.csv')
```

### Reading Common Biological File Formats
```python
import pandas as pd
import numpy as np

# CSV files (common lab data)
data = pd.read_csv('experiment_data.csv')

# Excel files
data = pd.read_excel('lab_results.xlsx')

# FASTA sequences
sequences = []
with open('sequences.fasta', 'r') as f:
    content = f.read()
    sequences = content.split('>')[1:]  # Simple FASTA parsing
```

---

## 🐍 Python Essentials for Biology

### Variables and Data Types
```python
# Numbers
cell_count = 50000
concentration = 2.5
ph_value = 7.4

# Text (strings)
gene_name = "BRCA1"
sequence = "ATCGATCG"

# Lists (collections)
samples = ["Control1", "Control2", "Treatment1"]
od_values = [0.1, 0.2, 0.8, 1.2]

# Check data type
type(cell_count)  # <class 'int'>
```

### String Operations for Sequences
```python
dna = "ATCGATCGTAGC"

# Basic operations
len(dna)              # Length: 12
dna.upper()           # Convert to uppercase
dna.lower()           # Convert to lowercase
dna.count('A')        # Count bases: 3

# Finding patterns
dna.find('ATG')       # Find start codon: 5
'ATG' in dna         # Check if present: True

# Slicing (extracting parts)
dna[0:3]             # First 3 bases: 'ATC'
dna[-3:]             # Last 3 bases: 'AGC'
dna[::3]             # Every 3rd base: 'ACGG'
```

### Lists for Data Collection
```python
# Creating lists
concentrations = [1.0, 2.5, 5.0, 10.0]
samples = ["S001", "S002", "S003"]

# Adding data
concentrations.append(15.0)
samples.extend(["S004", "S005"])

# Accessing elements
first_conc = concentrations[0]  # 1.0
last_sample = samples[-1]       # "S005"

# List operations
max(concentrations)    # Maximum value
min(concentrations)    # Minimum value
sum(concentrations)    # Sum of all values
len(samples)          # Number of elements
```

---

## 📈 Data Analysis with Pandas

### Creating DataFrames
```python
import pandas as pd

# From dictionary
data = {
    'Sample': ['S001', 'S002', 'S003'],
    'Treatment': ['Control', 'Drug A', 'Drug B'],
    'Cell_Count': [50000, 45000, 60000],
    'Viability': [95.2, 87.6, 91.3]
}
df = pd.DataFrame(data)

# From CSV file
df = pd.read_csv('experiment_data.csv')
```

### Basic DataFrame Operations
```python
# View data
df.head()           # First 5 rows
df.tail(3)          # Last 3 rows
df.info()           # Data types and info
df.describe()       # Statistical summary

# Selecting data
df['Sample']                    # Single column
df[['Sample', 'Cell_Count']]   # Multiple columns
df[df['Viability'] > 90]       # Filter rows

# Basic statistics
df['Cell_Count'].mean()        # Average
df['Cell_Count'].std()         # Standard deviation
df['Viability'].max()          # Maximum value
```

---

## 📊 Data Visualization

### Basic Plots with Matplotlib
```python
import matplotlib.pyplot as plt

# Line plot (time course)
time = [0, 2, 4, 6, 8, 24]
od600 = [0.1, 0.2, 0.5, 1.2, 1.8, 2.1]

plt.figure(figsize=(8, 6))
plt.plot(time, od600, marker='o')
plt.xlabel('Time (hours)')
plt.ylabel('OD600')
plt.title('Bacterial Growth Curve')
plt.grid(True)
plt.show()

# Bar plot (comparing treatments)
treatments = ['Control', 'Drug A', 'Drug B']
cell_counts = [50000, 45000, 60000]

plt.figure(figsize=(8, 6))
plt.bar(treatments, cell_counts)
plt.xlabel('Treatment')
plt.ylabel('Cell Count')
plt.title('Treatment Effect on Cell Count')
plt.show()

# Scatter plot (correlation)
plt.figure(figsize=(8, 6))
plt.scatter(concentration, viability)
plt.xlabel('Drug Concentration (μM)')
plt.ylabel('Cell Viability (%)')
plt.title('Dose-Response Curve')
plt.show()
```

### Advanced Plots with Seaborn
```python
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Box plot (comparing groups)
sns.boxplot(data=df, x='Treatment', y='Cell_Count')
plt.title('Cell Count by Treatment')
plt.show()

# Heatmap (correlation matrix)
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Variable Correlations')
plt.show()
```

---

## 🧬 Bioinformatics Examples

### DNA Sequence Analysis
```python
def analyze_sequence(dna):
    """Analyze DNA sequence composition"""
    dna = dna.upper()
    
    results = {
        'Length': len(dna),
        'A_count': dna.count('A'),
        'T_count': dna.count('T'),
        'G_count': dna.count('G'),
        'C_count': dna.count('C'),
    }
    
    # Calculate percentages
    total = results['Length']
    results['GC_content'] = ((results['G_count'] + results['C_count']) / total) * 100
    results['AT_content'] = ((results['A_count'] + results['T_count']) / total) * 100
    
    return results

# Example usage
sequence = "ATCGATCGTAGCTAGCTAGC"
analysis = analyze_sequence(sequence)
print(f"GC Content: {analysis['GC_content']:.1f}%")
```

### Processing Multiple Sequences
```python
sequences = [
    ("Gene1", "ATCGATCGTAGC"),
    ("Gene2", "GCGCGCGCGCGC"),
    ("Gene3", "ATATATATATATAT")
]

results = []
for name, seq in sequences:
    analysis = analyze_sequence(seq)
    results.append({
        'Gene': name,
        'Length': analysis['Length'],
        'GC_Content': analysis['GC_content']
    })

# Convert to DataFrame for easy viewing
results_df = pd.DataFrame(results)
print(results_df)
```

---

## 🔬 Statistical Analysis for Biology

### Basic Statistics
```python
import numpy as np
from scipy import stats

# Sample data
control = [23.1, 24.5, 22.8, 25.2, 23.7]
treatment = [28.3, 29.1, 27.8, 30.2, 28.9]

# Descriptive statistics
print(f"Control mean: {np.mean(control):.2f}")
print(f"Treatment mean: {np.mean(treatment):.2f}")
print(f"Control std: {np.std(control, ddof=1):.2f}")
print(f"Treatment std: {np.std(treatment, ddof=1):.2f}")

# T-test
t_stat, p_value = stats.ttest_ind(control, treatment)
print(f"T-test p-value: {p_value:.4f}")
```

### Working with Biological Data
```python
# IC50 calculation example
def calculate_ic50(concentrations, responses):
    """Simple IC50 estimation"""
    # Find concentration closest to 50% response
    target = 50.0
    differences = [abs(response - target) for response in responses]
    min_index = differences.index(min(differences))
    return concentrations[min_index]

# Example dose-response data
concentrations = [0.1, 1, 10, 100, 1000]  # μM
cell_viability = [95, 85, 65, 35, 15]     # %

ic50 = calculate_ic50(concentrations, cell_viability)
print(f"Estimated IC50: {ic50} μM")
```

---

## 🛠️ Useful Libraries for Biology

### Essential Imports
```python
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics
from scipy import stats
import statsmodels.api as sm

# Bioinformatics (if available)
# from Bio import SeqIO
# import requests  # For API calls
```

### Installing Additional Packages
```python
# Install packages not included in Colab
!pip install biopython
!pip install plotly

# Then import
from Bio import SeqIO
import plotly.express as px
```

---

## 🎯 Best Practices

### Code Organization
```python
# 1. Import all libraries at the top
import pandas as pd
import matplotlib.pyplot as plt

# 2. Define functions before using them
def calculate_fold_change(treatment, control):
    return treatment / control

# 3. Use descriptive variable names
protein_concentration = 2.5  # mg/mL
cell_viability_percent = 85.3

# 4. Add comments to explain complex operations
# Calculate growth rate using exponential growth formula
growth_rate = np.log(final_od / initial_od) / time_hours
```

### Data Management
```python
# Always check your data first
print(df.head())
print(df.info())

# Handle missing values
df = df.dropna()  # Remove rows with missing data
# or
df = df.fillna(0)  # Replace missing with 0

# Validate data ranges
assert all(df['pH'] >= 0), "pH values should be positive"
assert all(df['pH'] <= 14), "pH values should be <= 14"
```

### Reproducibility
```python
# Set random seeds for reproducible results
np.random.seed(42)

# Document your analysis steps
print("Step 1: Loading data...")
print("Step 2: Cleaning data...")
print("Step 3: Statistical analysis...")

# Save intermediate results
df.to_csv('processed_data.csv', index=False)
```

---

## 🚨 Common Errors and Solutions

### File Not Found Errors
```python
# Problem: File not found
# Solution: Check file path and existence
import os
if os.path.exists('data.csv'):
    df = pd.read_csv('data.csv')
else:
    print("File not found! Check the filename and location.")
```

### Index Errors
```python
# Problem: List index out of range
data = [1, 2, 3]
# data[5]  # This will cause an error

# Solution: Check list length first
if len(data) > 5:
    value = data[5]
else:
    print(f"List only has {len(data)} elements")
```

### Type Errors
```python
# Problem: Can't add string and number
# result = "Sample" + 123  # Error!

# Solution: Convert types
result = "Sample" + str(123)  # "Sample123"
# or use f-strings
result = f"Sample{123}"  # "Sample123"
```

---

## 🔍 Debugging Tips

### Print Statements
```python
# Use print to check variable values
concentration = 2.5
print(f"Concentration: {concentration}")

# Check data types
print(f"Type: {type(concentration)}")

# Check shape of arrays/DataFrames
print(f"Shape: {df.shape}")
```

### Error Messages
- **Read the error message carefully** - it tells you what went wrong
- **Look at the line number** - shows you where the error occurred  
- **Google the error** - others have likely had the same problem
- **Use try-except** for handling expected errors

```python
try:
    result = some_calculation()
except ZeroDivisionError:
    print("Cannot divide by zero!")
except ValueError:
    print("Invalid value provided!")
```

---

## 🎓 Learning Resources

### Documentation
- **Pandas**: pandas.pydata.org
- **Matplotlib**: matplotlib.org
- **NumPy**: numpy.org
- **Seaborn**: seaborn.pydata.org

### Biology-Specific Resources
- **Biopython Tutorial**: biopython.org/tutorial
- **Python for Biologists**: pythonforbiologists.com
- **Rosalind Problems**: rosalind.info

### Getting Help
```python
# Built-in help
help(pd.read_csv)
?pd.read_csv  # In Jupyter/Colab

# Online communities
# - Stack Overflow
# - Reddit r/bioinformatics
# - Biostars.org
```

---

## 📋 Quick Reference

### File Operations
```python
# Read files
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx')

# Save files  
df.to_csv('output.csv', index=False)
df.to_excel('output.xlsx', index=False)
```

### Data Selection
```python
# Columns
df['column_name']
df[['col1', 'col2']]

# Rows
df.iloc[0]        # First row
df.iloc[0:5]      # First 5 rows
df.loc[df['col'] > 5]  # Conditional selection
```

### Basic Statistics
```python
df.mean()         # Column means
df.std()          # Standard deviations
df.corr()         # Correlation matrix
df.describe()     # Summary statistics
```

### Plotting
```python
plt.plot(x, y)           # Line plot
plt.scatter(x, y)        # Scatter plot
plt.bar(categories, values)  # Bar plot
plt.hist(data)           # Histogram
plt.show()               # Display plot
```

---

*Remember: Google Colab auto-saves your work, but download important notebooks to your computer as backup!*

**Happy coding! 🧬💻**