"""
Interactive 3D Plotting GUI for Infomap Results

This module provides a GUI for visualizing Infomap clustering results in 3D
with interactive selection of neurotransmitter combinations and plotting parameters.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import os
import glob
from pathlib import Path
from ipywidgets import Checkbox, VBox, Button, Output, IntSlider, HBox, HTML, Dropdown, SelectMultiple
from IPython.display import display, clear_output


# Configure Plotly renderer for Jupyter
try:
    # Try different renderers until one works
    if 'notebook' in pio.renderers:
        pio.renderers.default = 'notebook'
    elif 'notebook_connected' in pio.renderers:
        pio.renderers.default = 'notebook_connected'
    elif 'jupyterlab' in pio.renderers:
        pio.renderers.default = 'jupyterlab'
    else:
        pio.renderers.default = 'iframe'
except:
    pass


def scan_available_results(base_dir=".", pattern_prefix="gui_output"):
    """
    Scan for all available Infomap results and return organized information.
    
    Returns:
    --------
    dict
        Dictionary with analysis results organized by neurotransmitter combination
    """
    results_info = {}
    
    # Find all matching output directories
    pattern = f"{pattern_prefix}_*_thresh*"
    output_dirs = glob.glob(os.path.join(base_dir, pattern))
    
    for output_dir in output_dirs:
        # Look for .tree and .net files
        tree_files = glob.glob(os.path.join(output_dir, "*.tree"))
        pajek_files = glob.glob(os.path.join(output_dir, "*.net"))
        
        if tree_files and pajek_files:
            # Extract combo_name and threshold from directory name
            dir_name = os.path.basename(output_dir)
            parts = dir_name.split('_')
            
            # Find threshold
            threshold = None
            combo_parts = []
            for part in parts[2:]:  # Skip "gui" and "output"
                if part.startswith('thresh'):
                    try:
                        threshold = int(part.replace('thresh', ''))
                    except ValueError:
                        threshold = 1
                else:
                    combo_parts.append(part)
            
            combo_name = "_".join(combo_parts)
            
            # Get file modification time for sorting
            mod_time = os.path.getmtime(output_dir)
            
            if combo_name not in results_info:
                results_info[combo_name] = []
            
            results_info[combo_name].append({
                'output_dir': output_dir,
                'tree_file': tree_files[0],
                'pajek_file': pajek_files[0],
                'combo_name': combo_name,
                'threshold': threshold if threshold else 1,
                'mod_time': mod_time,
                'dir_name': dir_name
            })
    
    # Sort each combo by threshold
    for combo in results_info:
        results_info[combo].sort(key=lambda x: x['threshold'])
    
    return results_info


def infer_neurotransmitters_from_combo(combo_name):
    """
    Infer neurotransmitter types from combo name.
    """
    nt_mapping = {
        'ach': 'acetylcholine',
        'acetylcholine': 'acetylcholine', 
        'gaba': 'GABA',
        'da': 'dopamine',
        'dopamine': 'dopamine',
        '5ht': 'serotonin',
        'serotonin': 'serotonin',
        'octopamine': 'octopamine',
        'glutamate': 'glutamate',
        'tyramine': 'tyramine',
        'glut': 'glutamate',
        'oct': 'octopamine',
        'ser': 'serotonin'
    }
    
    parts = combo_name.lower().split('_')
    neurotransmitters = []
    
    for part in parts:
        if part in nt_mapping:
            neurotransmitters.append(nt_mapping[part])
        else:
            # If not found in mapping, use the part as-is (capitalized)
            neurotransmitters.append(part.capitalize())
    
    return neurotransmitters


def parse_infomap_tree(tree_file):
    """
    Parse Infomap .tree output file to extract module information.
    """
    if not os.path.exists(tree_file):
        raise FileNotFoundError(f"Tree file not found: {tree_file}")
    
    modules = []
    
    with open(tree_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(' ')
                if len(parts) >= 4:
                    try:
                        module_path = parts[0]
                        flow = float(parts[1])
                        node_name = parts[2]
                        node_id = parts[3] if len(parts) > 3 else node_name
                        
                        # Extract top-level module ID
                        module_id = module_path.split(':')[0]
                        
                        modules.append({
                            'module_id': int(module_id),
                            'node_id': node_id,
                            'node_name': node_name,
                            'flow': flow,
                            'full_path': module_path
                        })
                    except (ValueError, IndexError):
                        continue
    
    return pd.DataFrame(modules)


def create_3d_plot(results_info, coords_file, connections_file, top_n_modules):
    """
    Create the actual 3D plot from results.
    """
    tree_file = results_info['tree_file']
    pajek_file = results_info['pajek_file']
    combo_name = results_info['combo_name']
    threshold = results_info['threshold']
    
    # ====== STEP 1: Rebuild node ID mapping from Pajek ======
    id_to_node = {}
    with open(pajek_file, "r") as f:
        for line in f:
            if line.startswith("*Vertices"):
                continue
            if line.startswith("*Edges") or line.startswith("*Arcs"):
                break
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    id_num = int(parts[0])
                    root_id = parts[1].strip('"')
                    id_to_node[id_num] = root_id
                except (ValueError, IndexError):
                    continue

    # ====== STEP 2: Load tree and parse module info ======
    tree_data = []
    with open(tree_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    module_path = parts[0]
                    node_index = int(parts[-1])
                    root_id = str(id_to_node.get(node_index, "UNKNOWN"))
                    tree_data.append((root_id, module_path))
                except (ValueError, IndexError):
                    continue

    if not tree_data:
        raise ValueError("No valid tree data found")

    modules_df = pd.DataFrame(tree_data, columns=["neuron_id", "module_path"])
    modules_df["top_module"] = modules_df["module_path"].str.split(":").str[0].astype(int)

    # ====== STEP 3: Load coordinates and merge ======
    if not os.path.exists(coords_file):
        raise FileNotFoundError(f"Coordinates file not found: {coords_file}")
    
    coords_df = pd.read_csv(coords_file)
    
    # Handle different coordinate file formats
    if "root_id" in coords_df.columns:
        coords_df["neuron_id"] = coords_df["root_id"].astype(str)
    elif "neuron_id" in coords_df.columns:
        coords_df["neuron_id"] = coords_df["neuron_id"].astype(str)
    else:
        id_cols = [col for col in coords_df.columns if 'id' in col.lower()]
        if id_cols:
            coords_df["neuron_id"] = coords_df[id_cols[0]].astype(str)
        else:
            raise ValueError("Could not find neuron ID column in coordinates file")
    
    merged = pd.merge(modules_df, coords_df, on="neuron_id", how="inner")
    
    if len(merged) == 0:
        raise ValueError("No neurons matched between tree and coordinates")

    # ====== STEP 4: Limit to top N modules ======
    module_counts = merged["top_module"].value_counts()
    top_modules = module_counts.head(top_n_modules).index.tolist()
    top_merged = merged[merged["top_module"].isin(top_modules)].copy()

    # ====== STEP 5: Parse coordinates ======
    if "position" in top_merged.columns:
        def parse_position(pos_str):
            try:
                if isinstance(pos_str, str):
                    coords = pos_str.strip("[]").split()
                    if len(coords) >= 3:
                        return pd.Series([float(coords[0]), float(coords[1]), float(coords[2])])
                return pd.Series([np.nan, np.nan, np.nan])
            except:
                return pd.Series([np.nan, np.nan, np.nan])
        
        top_merged[["x", "y", "z"]] = top_merged["position"].apply(parse_position)
    
    elif all(col in top_merged.columns for col in ["x", "y", "z"]):
        top_merged[["x", "y", "z"]] = top_merged[["x", "y", "z"]].astype(float)
    
    else:
        coord_cols = []
        for potential in [["x", "y", "z"], ["X", "Y", "Z"], ["pos_x", "pos_y", "pos_z"]]:
            if all(col in top_merged.columns for col in potential):
                coord_cols = potential
                break
        
        if coord_cols:
            top_merged[["x", "y", "z"]] = top_merged[coord_cols].astype(float)
        else:
            raise ValueError("Could not find x, y, z coordinates")
    
    # Remove rows with invalid coordinates
    valid_coords = ~(top_merged[["x", "y", "z"]].isna().any(axis=1))
    top_merged = top_merged[valid_coords].copy()
    
    if len(top_merged) == 0:
        raise ValueError("No neurons have valid coordinates")

        # ====== STEP 6: Compute degree (network connectivity) ======
    if os.path.exists(connections_file):
        try:
            conn_df = pd.read_csv(connections_file)

            # —— Normalize nt_type to lowercase so matching is reliable —— 
            if 'nt_type' in conn_df.columns:
                conn_df['nt_type'] = conn_df['nt_type'].str.lower()
                # also lowercase your inferred list
                selected_nts = [nt.lower() for nt in infer_neurotransmitters_from_combo(combo_name)]
                conn_df = conn_df[conn_df['nt_type'].isin(selected_nts)]

            # Ensure IDs are strings
            conn_df["pre_root_id"] = conn_df["pre_root_id"].astype(str)
            conn_df["post_root_id"] = conn_df["post_root_id"].astype(str)

            # Keep only edges touching your clustered neurons
            valid_neurons = set(top_merged["neuron_id"])
            conn_df = conn_df[
                conn_df["pre_root_id"].isin(valid_neurons) |
                conn_df["post_root_id"].isin(valid_neurons)
            ]

            # Melt in- and out-edges into one column
            all_ends = pd.concat([
                conn_df[["pre_root_id"]].rename(columns={"pre_root_id": "neuron_id"}),
                conn_df[["post_root_id"]].rename(columns={"post_root_id": "neuron_id"})
            ], ignore_index=True)

            # Count degree per neuron
            degree_counts = (
                all_ends
                  .groupby("neuron_id")
                  .size()
                  .reset_index(name="degree")
            )

            # Merge back, fill missing as zero
            top_merged = top_merged.merge(degree_counts, on="neuron_id", how="left")
            top_merged["degree"] = top_merged["degree"].fillna(0).astype(int)

        except Exception:
            # fallback if something goes wrong
            top_merged["degree"] = 1
    else:
        top_merged["degree"] = 1

    # ====== STEP 7: Create 3D plot ======
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', 
              '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7', 
              '#dbdb8d', '#9edae5']
    
    for i, module in enumerate(sorted(top_merged["top_module"].unique())):
        cluster = top_merged[top_merged["top_module"] == module]
        color = colors[i % len(colors)]
        
        # Scale point sizes based on degree
        if top_merged["degree"].max() > 0:
            sizes = np.log1p(cluster["degree"]) / np.log1p(top_merged["degree"].max()) * 15 + 3        
        else:
            sizes = [8] * len(cluster)
        
        fig.add_trace(go.Scatter3d(
            x=cluster["x"],
            y=cluster["y"], 
            z=cluster["z"],
            mode='markers',
            marker=dict(
                size=sizes,
                color=color,
                opacity=0.1
            ),
            name=f"Module {module} ({len(cluster)} neurons)",
            text=[f"Neuron: {nid}<br>Module: {module}<br>Degree: {deg:.0f}" 
                  for nid, deg in zip(cluster["neuron_id"], cluster["degree"])],
            hovertemplate="<b>%{text}</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>"
        ))

    # Update layout
    nt_display = ", ".join(infer_neurotransmitters_from_combo(combo_name))
    fig.update_layout(
        title={
            'text': f"Top {top_n_modules} Infomap Modules: {nt_display} (threshold ≥{threshold})",
            'x': 0.5,
            'font': {'size': 16}
        },
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate', 
            zaxis_title='Z Coordinate',
            camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)),
            aspectratio=dict(x=2, y=1, z=1)
        ),
        legend=dict(
            title="Infomap Modules",
            x=1.02, y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        width=1000, height=750,
        margin=dict(l=0, r=150, b=0, t=50),
        font=dict(size=12)
    )

    return fig, len(top_merged), len(top_modules)


def display_plot_safely(fig, output_widget):
    """
    Try multiple methods to display the plot.
    """
    try:
        # Method 1: Direct show
        with output_widget:
            fig.show()
            return True
    except Exception as e1:
        try:
            # Method 2: Show with specific renderer
            with output_widget:
                fig.show(renderer='notebook')
                return True
        except Exception as e2:
            try:
                # Method 3: Show as HTML
                with output_widget:
                    html_str = fig.to_html(include_plotlyjs='cdn')
                    from IPython.display import HTML
                    display(HTML(html_str))
                    return True
            except Exception as e3:
                # Method 4: Save and show file path
                with output_widget:
                    try:
                        filename = "temp_plot.html"
                        fig.write_html(filename)
                        print(f"Plot saved as '{filename}' - open this file in your browser")
                        return True
                    except Exception as e4:
                        print(f"   All display methods failed:")
                        print(f"   Direct show: {e1}")
                        print(f"   Notebook renderer: {e2}")
                        print(f"   HTML display: {e3}")
                        print(f"   File save: {e4}")
                        return False


def create_plotting_gui(coords_file="coordinates.csv", connections_file="connections_princeton.csv"):
    """
    Create an interactive GUI for plotting Infomap analysis results in 3D.
    
    This GUI automatically scans for available Infomap results and allows users to:
    - Select which neurotransmitter combination to visualize
    - Choose the synapse threshold
    - Set the number of modules to display
    - Generate interactive 3D plots with one click
    
    Parameters:
    -----------
    coords_file : str, default="coordinates.csv"
        Path to the neuron coordinates CSV file
    connections_file : str, default="connections_princeton.csv"
        Path to the connections CSV file (for degree calculations)
        
    Example:
    --------
    >>> create_plotting_gui()
    >>> # Or with custom files:
    >>> create_plotting_gui("my_coords.csv", "my_connections.csv")
    """
    
    # Scan for available results
    try:
        available_results = scan_available_results()
        if not available_results:
            display(HTML("""
                <div style='color: red; font-size: 14px; padding: 20px; border: 2px solid red; border-radius: 5px; background: #ffebee;'>
                <b>No Infomap Results Found</b><br>
                No analysis results were found. Please run the analysis GUI first to generate Infomap results.
                </div>
            """))
            return
    except Exception as e:
        display(HTML(f"""
            <div style='color: red; font-size: 14px; padding: 20px; border: 2px solid red; border-radius: 5px; background: #ffebee;'>
            <b>  Error Scanning Results</b><br>
            Could not scan for available results: {e}
            </div>
        """))
        return
    
    # ====== CREATE GUI COMPONENTS ======
    
    # Results summary
    total_combos = len(available_results)
    total_analyses = sum(len(results) for results in available_results.values())
    
    summary_html = HTML(f"""
        <div style='font-size: 12px; color: #2e7d32; margin: 10px 0; padding: 10px; background: #e8f5e8; border-left: 4px solid #4caf50; border-radius: 0 5px 5px 0;'>
        <b> Found {total_analyses} analysis results across {total_combos} neurotransmitter combinations</b>
        </div>
    """)
    
    # Neurotransmitter selection
    combo_options = []
    combo_data = {}
    
    for combo_name, results_list in available_results.items():
        # Create display name
        nt_names = infer_neurotransmitters_from_combo(combo_name)
        display_name = " + ".join(nt_names)
        
        # Get available thresholds
        thresholds = [r['threshold'] for r in results_list]
        thresh_str = f"(thresh: {', '.join(map(str, thresholds))})"
        
        full_display = f"{display_name} {thresh_str}"
        combo_options.append(full_display)
        combo_data[full_display] = results_list
    
    combo_dropdown = Dropdown(
        options=combo_options,
        value=combo_options[0] if combo_options else None,
        description='NT Combination:',
        style={'description_width': '120px'},
        layout={'width': '500px'}
    )
    
    # Threshold selection (will be updated based on combo selection)
    threshold_dropdown = Dropdown(
        options=[],
        description='Threshold:',
        style={'description_width': '120px'},
        layout={'width': '200px'}
    )
    
    # Number of modules slider
    modules_slider = IntSlider(
        value=15,
        min=5,
        max=50,
        step=1,
        description='Top N Modules:',
        style={'description_width': '120px'},
        layout={'width': '400px'}
    )
    
    # File status
    file_status_html = HTML()
    
    def update_file_status():
        coord_exists = os.path.exists(coords_file)
        conn_exists = os.path.exists(connections_file)
        
        coord_icon = "✅" if coord_exists else "❌"
        conn_icon = "✅" if conn_exists else "❌"
        
        file_status_html.value = f"""
        <div style='font-size: 11px; color: #666; margin: 10px 0; padding: 8px; background: #f5f5f5; border-radius: 3px;'>
        <b>📁 File Status:</b><br>
        {coord_icon} Coordinates: {coords_file}<br>
        {conn_icon} Connections: {connections_file}
        </div>
        """
    
    update_file_status()
    
    # Analysis info display
    analysis_info_html = HTML()
    
    # Update threshold dropdown when combo selection changes
    def update_threshold_options(change):
        selected_combo = combo_dropdown.value
        if selected_combo and selected_combo in combo_data:
            results_list = combo_data[selected_combo]
            threshold_options = [(f"≥{r['threshold']} synapses", r) for r in results_list]
            threshold_dropdown.options = threshold_options
            threshold_dropdown.value = threshold_options[0][1] if threshold_options else None
            
            # Update analysis info
            if threshold_dropdown.value:
                update_analysis_info()
    
    def update_analysis_info(change=None):
        if threshold_dropdown.value:
            result_info = threshold_dropdown.value
            
            # Get module count from tree file
            try:
                tree_df = parse_infomap_tree(result_info['tree_file'])
                num_modules = tree_df['module_id'].nunique()
                num_neurons = len(tree_df)
                
                analysis_info_html.value = f"""
                <div style='font-size: 12px; color: #1565c0; margin: 10px 0; padding: 10px; background: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 0 5px 5px 0;'>
                <b>Selected Analysis:</b><br>
                • Directory: {os.path.basename(result_info['output_dir'])}<br>
                • Modules found: {num_modules}<br>
                • Neurons clustered: {num_neurons:,}<br>
                • Threshold: ≥{result_info['threshold']} synapses
                </div>
                """
            except Exception as e:
                analysis_info_html.value = f"""
                <div style='font-size: 12px; color: #d32f2f; margin: 10px 0; padding: 10px; background: #ffebee; border-left: 4px solid #f44336; border-radius: 0 5px 5px 0;'>
                <b>  Could not read analysis info:</b><br>
                {str(e)}
                </div>
                """
    
    combo_dropdown.observe(update_threshold_options, names='value')
    threshold_dropdown.observe(update_analysis_info, names='value')
    
    # Initialize threshold options
    if combo_options:
        update_threshold_options({'new': combo_dropdown.value})
    
    # Plot button and output
    plot_button = Button(
        description="Generate 3D Plot",
        button_style='success',
        layout={'width': '200px', 'height': '40px'}
    )
    
    output = Output()
    
    # Plot button callback
    from IPython.display import display
    
    def on_plot_clicked(b):
        # 1️⃣ Clear output once
        output.clear_output(wait=True)
    
        # 2️⃣ Validate selection
        if not threshold_dropdown.value:
            with output:
                print("Please select a neurotransmitter combination and threshold.")
            return
    
        result_info = threshold_dropdown.value
        top_n = modules_slider.value
    
        # 3️⃣ Print status up front
        with output:
            print(f"Generating 3D plot for {combo_dropdown.value}")
            print(f"Threshold: ≥{result_info['threshold']} synapses")
            print(f"Top {top_n} modules")
            print("─" * 40)
            print("Loading analysis results…")
    
        try:
            # 4️⃣ Build the figure
            fig, num_neurons, num_modules_plotted = create_3d_plot(
                result_info, coords_file, connections_file, top_n
            )
    
            # 5️⃣ Display the figure (no further clear_output calls)
            with output:
                print(f"Plot created! {num_neurons:,} neurons across {num_modules_plotted} modules")
                display(fig)
    
        except Exception as e:
            with output:
                print(f"Error creating plot: {e}")
                print("\n Troubleshooting:")
                print("• Check that coordinate and connection files exist and are readable")
                print("• Verify that the analysis results are not corrupted")
                print("• Try with a smaller number of modules")
    
    plot_button.on_click(on_plot_clicked)
    
    # Help section
    help_html = HTML(f"""
        <div style='font-size: 11px; color: #666; margin: 15px 0; padding: 8px; background: #f9f9f9; border-left: 3px solid #ddd;'>
        <b> Usage Tips:</b><br>
        • Select a neurotransmitter combination from the dropdown<br>
        • Choose the synapse threshold (higher = sparser network)<br>
        • Adjust the number of modules to display (5-50)<br>
        • Click "Generate 3D Plot" to create the interactive visualization<br>
        • Point sizes reflect network connectivity (degree)<br>
        • Colors distinguish different modules found by Infomap<br>
        <br>
        <b>Current Plotly Renderer:</b> {pio.renderers.default}
        </div>
    """)
    
    # Layout
    main_layout = VBox([
        HTML("<h2 style='color: #333; margin-bottom: 5px;'>3D Infomap Visualization</h2>"),
        HTML("<hr style='margin: 5px 0 15px 0;'>"),
        
        summary_html,
        file_status_html,
        
        HTML("<h3 style='color: #555; margin: 15px 0 5px 0;'>Analysis Selection</h3>"),
        HBox([combo_dropdown, threshold_dropdown]),
        analysis_info_html,
        
        HTML("<h3 style='color: #555; margin: 15px 0 5px 0;'>Plot Parameters</h3>"),
        modules_slider,
        
        help_html,
        
        HTML("<hr style='margin: 15px 0 10px 0;'>"),
        plot_button,
        HTML("<div style='margin-top: 10px;'>"),
        output,
        HTML("</div>")
    ])
    
    display(main_layout)