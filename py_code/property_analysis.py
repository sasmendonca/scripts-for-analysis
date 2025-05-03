import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from pathlib import Path
from reportlab.lib.units import inch
from PIL import Image as PILImage  # Import Pillow's Image module

def load_and_process_data(file_path, file_type='sdf'):
    """Loads and processes data from SDF, CSV, or Excel files."""
    if file_type == 'sdf':
        df = PandasTools.LoadSDF(file_path, smilesName='smiles', includeFingerprints=False)
    elif file_type == 'csv':
        df = pd.read_csv(file_path)
    elif file_type == 'excel':
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Use 'sdf', 'csv', or 'excel'.")
    
    # Initial processing
    if "pIC50" in df.columns:
        df["pIC50"] = df["pIC50"].astype(float)
        df["pIC50"] = [round(x, 2) for x in df["pIC50"]]
    if 'ROMol' in df.columns:
        df = df.drop(['ROMol'], axis=1)
    return df

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def calculate_descriptors(df, smiles_col='smiles'):
    """Calculates molecular descriptors for each molecule in the DataFrame."""
    def _calculate_descriptors(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return {
                'MW': rdMolDescriptors.CalcExactMolWt(mol),
                'LogP': rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
                'HBD': rdMolDescriptors.CalcNumHBD(mol),
                'HBA': rdMolDescriptors.CalcNumHBA(mol),
                'TPSA': rdMolDescriptors.CalcTPSA(mol),
                'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'RingCount': rdMolDescriptors.CalcNumRings(mol),
                'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
                'HeavyAtomCount': rdMolDescriptors.CalcNumHeavyAtoms(mol),
            }
        return {}
      
    def _calculate_solubility(smiles):
        """
        Calculate ESOL descriptors and solubility for a single SMILES.
        :param smiles: SMILES string
        :return: Dictionary with ESOL descriptors and solubility
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw = rdMolDescriptors.CalcExactMolWt(mol)
            logp = rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
            rotors = rdMolDescriptors.CalcNumRotatableBonds(mol)
            aromatic_query = Chem.MolFromSmarts('a') 
            matches = mol.GetSubstructMatches(aromatic_query)
            ap = len(matches) / mol.GetNumAtoms()
            esol = -0.01 * mw + 0.54 * logp - 0.63 * rotors + 0.06 * ap - 0.5
            return {'ESOL_LogS': esol}

    def _calculate_all_descriptors(smiles):
        """
        Combine molecular descriptors and ESOL descriptors for a single SMILES.
        :param smiles: SMILES string
        :return: Combined dictionary of descriptors
        """
        descriptors = _calculate_descriptors(smiles)
        esol_descriptors = _calculate_solubility(smiles)
        return {**(descriptors if isinstance(descriptors, dict) else {}), 
                **(esol_descriptors if isinstance(esol_descriptors, dict) else {})}

    logger.info("Calculating molecular descriptors and ESOL descriptors...")
    all_descriptors = df[smiles_col].apply(_calculate_all_descriptors).apply(pd.Series)

    return pd.concat([df, all_descriptors], axis=1)

def generate_plots(df, save_path=None):
    """Generates data visualizations and optionally saves them to a file."""
    logger.info("Generating property analysis plots...")
    
    # Configure layout to display two plots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.set(style="whitegrid")
    ax1, ax2 = axes if isinstance(axes, np.ndarray) else (axes, axes)
    
    # Histogram using seaborn
    sns.histplot(df["pIC50"].dropna(), bins=100, color='green', alpha=0.75, ax=ax1)
    ax1.set_xlabel('Measured pIC50', fontsize=14)
    ax1.set_ylabel('Number of compounds', fontsize=14)
    ax1.set_title('Histogram of pIC50 Values')

    # Distribution with KDE using seaborn
    sns.histplot(df["pIC50"].dropna(), kde=True, color='blue', bins=30, ax=ax2)
    ax2.set_xlabel('pIC50', fontsize=14)
    ax2.set_ylabel('Density', fontsize=14)
    ax2.set_title('Distribution of pIC50 Values')

    plt.tight_layout()
    plot_paths = []
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plot_paths.append(save_path)
    plt.close()

    # FacetGrid for descriptors vs pIC50 (Part 1)
    descriptors = ['MW', 'LogP', 'TPSA', 'ESOL_LogS']
    valid_descriptors = [desc for desc in descriptors if desc in df.columns]
    
    if valid_descriptors:
        melted_df = df.melt(id_vars='pIC50', value_vars=valid_descriptors,
                            var_name='Descriptor', value_name='Value')
        g = sns.FacetGrid(melted_df, col="Descriptor", col_wrap=2, 
                         height=4, sharex=False, sharey=False)
        g.map(sns.scatterplot, "pIC50", "Value", alpha=0.7, color='blue')
        g.set_titles("{col_name} vs pIC50")
        g.set_axis_labels("pIC50", "Descriptor Value")
        plt.tight_layout()
        
        if save_path:
            facet_path1 = save_path.replace('.png', '_facet1.png')
            g.savefig(facet_path1, dpi=150, bbox_inches='tight')
            plot_paths.append(facet_path1)
        plt.close()

    # FacetGrid for descriptors vs pIC50 (Part 2)
    descriptors = ['HBD', 'HBA', 'NumRotatableBonds', 'RingCount']
    valid_descriptors = [desc for desc in descriptors if desc in df.columns]
    
    if valid_descriptors:
        melted_df = df.melt(id_vars='pIC50', value_vars=valid_descriptors,
                            var_name='Descriptor', value_name='Value')
        g = sns.FacetGrid(melted_df, col="Descriptor", col_wrap=2,
                         height=4, sharex=False, sharey=False)
        g.map(sns.scatterplot, "pIC50", "Value", alpha=0.7, color='blue')
        g.set_titles("{col_name} vs pIC50")
        g.set_axis_labels("pIC50", "Descriptor Value")
        plt.tight_layout()
        
        if save_path:
            facet_path2 = save_path.replace('.png', '_facet2.png')
            g.savefig(facet_path2, dpi=150, bbox_inches='tight')
            plot_paths.append(facet_path2)
        plt.close()

    return plot_paths

def generate_report(df, filename='analytical_report.pdf', report_path='./output/'):
    """
    Generates a professional PDF report with:
    - Descriptive statistics
    - Basic statistics graph
    - Distribution chart
    - Plots from generate_plots
    - Correlation analysis
    """
    
    # Configure full path
    Path(report_path).mkdir(parents=True, exist_ok=True)
    full_path = Path(report_path) / filename
    doc = SimpleDocTemplate(str(full_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    temp_files = []  # List to store temporary files
    
    # Create document
    doc = SimpleDocTemplate(str(full_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # --- Section 1: Header ---
    title = Paragraph("Comprehensive Analytical Report for " + Path(filename).stem, styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # --- Section 2: Descriptive Statistics ---
    story.append(Paragraph("Basic Statistics", styles['Heading2']))
    
    # Adjust table to fit A4 page width
    stats = df.describe().reset_index()
    stats_data = [stats.columns.values.tolist()] + stats.values.tolist()
    
    # Calculate column widths dynamically based on the number of columns
    num_columns = len(stats.columns)
    column_width = 7.0 * inch / num_columns  # Adjust to fit within A4 width (7.0 inches for margins)
    col_widths = [column_width] * num_columns
    
    # Format numeric values and wrap column headers for clarity
    formatted_stats_data = []
    for i, row in enumerate(stats_data):
        if i == 0:  # Header row
            formatted_row = [Paragraph("<br/>".join(x.split()), styles['Normal']) if isinstance(x, str) else x for x in row]
        else:  # Data rows
            formatted_row = [f"{x:.2f}" if isinstance(x, (int, float)) else x for x in row]
        formatted_stats_data.append(formatted_row)
    
    # Create formatted table with adjusted column widths
    stats_table = Table(formatted_stats_data, colWidths=col_widths, repeatRows=1)
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTSIZE', (0,1), (-1,-1), 8),  # Smaller font size for data rows
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')  # Vertically align text in the middle
    ]))
    
    # Split the table into smaller chunks if it doesn't fit on one page
    if len(stats_data) > 20:  # Adjust the threshold as needed
        for i in range(0, len(stats_data), 20):  # Split every 20 rows
            chunk = stats_data[i:i + 20]
            chunk_table = Table(chunk, colWidths=col_widths, repeatRows=1)
            chunk_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTSIZE', (0,0), (-1,0), 10),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.white),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('FONTSIZE', (0,1), (-1,-1), 8),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
            ]))  # Reuse the same style
            story.append(chunk_table)
            story.append(PageBreak())
    else:
        story.append(stats_table)
    story.append(Spacer(1, 24))
    
    # Add violin boxplot for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if not numeric_columns.empty:
        for column in numeric_columns:
            plt.figure(figsize=(6, 4))
            sns.violinplot(y=column, data=df, inner='box', palette='muted')
            plt.title(f'Violin Boxplot for {column}', fontsize=14)
            violin_plot_path = Path(report_path) / f'temp_violin_plot_{column}.png'
            violin_plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(violin_plot_path), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Add violin plot to the PDF
            violin_img = Image(violin_plot_path, width=4.5*inch, height=3*inch)
            story.append(Paragraph(f"Violin Boxplot for {column}", styles['Heading2']))
            story.append(Spacer(1, 12))
            story.append(violin_img)
            story.append(Spacer(1, 24))
        
        
    # --- Section 4: Plots from generate_plots ---
    # Property Analysis Section
    story.append(Paragraph("Property Analysis", styles['Heading2']))
    
    # Generate and add plots
    plot_save_path = Path(report_path) / 'temp_plots.png'
    plot_paths = generate_plots(df, save_path=str(plot_save_path))
    
    for plot_path in plot_paths:
        if plot_path and Path(plot_path).is_file():
            try:
                # Adjust image size
                try:
                    # Open the image to get its dimensions
                    with PILImage.open(plot_path) as img_obj:
                        width, height = img_obj.size
                        aspect_ratio = width / height
                        
                        # Adjust the size while maintaining the aspect ratio
                        max_width = 6 * inch
                        max_height = 4.5 * inch
                        if aspect_ratio > 1:  # Wider than tall
                            img_width = min(max_width, width * inch / 100)
                            img_height = img_width / aspect_ratio
                        else:  # Taller than wide
                            img_height = min(max_height, height * inch / 100)
                            img_width = img_height * aspect_ratio
                        
                        img = Image(plot_path, width=img_width, height=img_height)
                        story.append(img)
                        story.append(Spacer(1, 12))
                        temp_files.append(Path(plot_path))
                except Exception as e:
                    logger.error(f"Error adjusting plot size for {plot_path}: {e}")
            except Exception as e:
                logger.error(f"Error adding plot {plot_path}: {e}")
          
            
    # --- Section 5: Correlations ---
    story.append(Paragraph("Key Correlations", styles['Heading2']))
    
    # Calculate correlations
    numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
    corr_matrix = numeric_df.corr().round(2)
    corr_data = [[''] + corr_matrix.columns.tolist()]  # Header
    
    for col in corr_matrix.columns:
        row = [col] + corr_matrix[col].values.tolist()
        corr_data.append(row)
    
    # Wrap column headers for clarity
    formatted_corr_data = []
    for i, row in enumerate(corr_data):
        if i == 0:  # Header row
            formatted_row = [Paragraph("<in/>".join(str(x).split()), styles['Normal']) if isinstance(x, str) else x for x in row]
        else:  # Data rows
            formatted_row = row
        formatted_corr_data.append(formatted_row)
    
    # Correlation table
    corr_table = Table(formatted_corr_data)
    corr_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightblue),  # Lighter color for the header
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')  # Vertically align text in the middle
    ]))
    
    # Wrap text in the first column for better readability
    for i, row in enumerate(formatted_corr_data):
        if i > 0:  # Skip the header row
            row[0] = Paragraph("<in/>".join(str(row[0]).split()), styles['Normal'])
    
    story.append(corr_table)
    story.append(Spacer(1, 24))
    
    # Correlation heatmap plot
    plt.figure(figsize=(8, 6))  # Adjust figure size for better aspect ratio
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True)
    plt.title('Correlation Heatmap', fontsize=14)
    heatmap_path = Path(report_path) / 'temp_heatmap.png'
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    plt.savefig(str(heatmap_path), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Add heatmap to PDF with adjusted size and alignment
    heatmap_img = Image(heatmap_path, width=6*inch, height=4.5*inch)  # Adjust width and height for better proportions
    story.append(Paragraph("Correlation Heatmap", styles['Heading2']))
    story.append(Spacer(1, 12))
    story.append(heatmap_img)
    story.append(Spacer(1, 24))
    
    # --- Generate PDF ---
    doc.build(story)
    
    # Clean up temporary heatmap file
    Path(heatmap_path).unlink(missing_ok=True)
    
    # Clean up temporary plot files
    for temp_file in temp_files:
        temp_file.unlink(missing_ok=True)

    # Clean up temporary violin plot files
    for column in numeric_columns:
        violin_plot_path = Path(report_path) / f'temp_violin_plot_{column}.png'
        if violin_plot_path.exists():
            violin_plot_path.unlink(missing_ok=True)

    return full_path