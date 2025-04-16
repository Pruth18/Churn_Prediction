import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
def explain_with_shap(model: Pipeline, X, use_streamlit=False, st=None):
    """Generate SHAP summary and force plots for model explainability. If use_streamlit is True, display in Streamlit with interactive force plot if possible."""
    trained_model = model.named_steps['model']
    if hasattr(trained_model, 'feature_importances_') or 'XGB' in str(type(trained_model)):
        explainer = shap.TreeExplainer(trained_model)
    else:
        explainer = shap.Explainer(trained_model, X)
    X_sample = X.sample(min(1000, len(X)), random_state=42)
    shap_values = explainer.shap_values(X_sample)
    # Check for empty or invalid SHAP values
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_values_plot = shap_values
    if X_sample.shape[0] == 0 or shap_values_plot is None or (hasattr(shap_values_plot, 'size') and shap_values_plot.size == 0):
        if use_streamlit and st:
            st.warning("No SHAP values to plot.")
        else:
            print("No SHAP values to plot.")
        return
    # SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_plot, X_sample, plot_type="bar", show=False)
    plt.title('Feature Importance using SHAP Values')
    plt.tight_layout()
    if st is not None:
        st.pyplot(plt.gcf())
        plt.close()
    else:
        # Removed plt.show() to prevent non-interactive backend warning
        plt.close()
    # SHAP force plot
    if use_streamlit and st:
        # Always use the static matplotlib version for consistency
        st.markdown('## SHAP Force Plot')
        _show_static_top_features_force_plot(explainer, shap_values_plot, X_sample, st)
    else:
        _show_static_top_features_force_plot(explainer, shap_values_plot, X_sample)

def _show_static_top_features_force_plot(explainer, shap_values_plot, X_sample, st=None, top_n=6):
    """Create a cleaner, more readable SHAP force plot using matplotlib."""
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # Get the SHAP values and feature values for the sample
    shap_vals = shap_values_plot[0]
    feature_vals = X_sample.iloc[0]
    
    # Get indices of top features by absolute SHAP value
    top_idx = np.argsort(np.abs(shap_vals))[-top_n:]
    
    # Create subset with only top features
    top_feature_names = X_sample.columns[top_idx]
    top_shap_values = shap_vals[top_idx]
    top_feature_values = feature_vals.iloc[top_idx]
    
    # Create a cleaner display of feature names and values
    feature_labels = []
    for fname, fval in zip(top_feature_names, top_feature_values):
        # Format numeric values to 2 decimal places
        if isinstance(fval, (int, float)):
            if fval == int(fval):
                fval_str = str(int(fval))
            else:
                fval_str = f"{fval:.2f}"
        else:
            fval_str = str(fval)
            
        # Clean up feature name for display
        display_name = fname.replace('_', ' ').title()
        feature_labels.append(f"{display_name} = {fval_str}")
    
    # Create figure with larger size for better readability
    plt.figure(figsize=(12, 5))
    
    # Sort for better visualization
    sorted_indices = np.argsort(top_shap_values)
    sorted_shap_values = top_shap_values[sorted_indices]
    sorted_labels = [feature_labels[i] for i in sorted_indices]
    
    # Create improved colors for bars with gradient
    colors = []
    for x in sorted_shap_values:
        if x < 0:
            # Darker red for more negative values
            intensity = min(0.8, 0.4 + abs(x) / max(0.1, max(abs(sorted_shap_values))))
            colors.append((1.0, 0.3*intensity, 0.3*intensity))
        else:
            # Darker blue for more positive values
            intensity = min(0.8, 0.4 + abs(x) / max(0.1, max(abs(sorted_shap_values))))
            colors.append((0.3*intensity, 0.3*intensity, 1.0))
    
    # Create the bar chart with styled bars
    y_pos = np.arange(len(sorted_labels))
    bars = plt.barh(y_pos, sorted_shap_values, color=colors, edgecolor='#444444', alpha=0.9, height=0.6)
    
    # Add background shading to distinguish positive/negative regions
    plt.axvspan(0, max(0.1, max(sorted_shap_values)*1.1), alpha=0.05, color='blue')
    plt.axvspan(min(sorted_shap_values)*1.1, 0, alpha=0.05, color='red')
    plt.axvline(x=0, color='#444444', linestyle='-', alpha=0.5, linewidth=1.5)
    
    # Add feature labels with more space
    plt.yticks(y_pos, sorted_labels, fontsize=10)
    plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=11)
    
    # Add values on the bars
    for bar, value in zip(bars, sorted_shap_values):
        text_x = bar.get_width() * 1.01 if value >= 0 else bar.get_width() * 0.99
        ha = 'left' if value >= 0 else 'right'
        plt.text(text_x, bar.get_y() + bar.get_height()/2, f"{value:.3f}", 
                 va='center', ha=ha, fontsize=9, fontweight='bold')
    
    # Add base value annotation with better styling
    base_value = explainer.expected_value
    plt.figtext(0.5, 0.01, f"Base Value: {base_value:.3f} (average model output)", 
              fontsize=10, ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.5', 
                                                           facecolor='white', alpha=0.8))
    
    # Add a cleaner title and legend
    plt.title(f'Feature Impact on Prediction (Top {top_n} Features)', fontsize=12, pad=10)
    plt.figtext(0.5, 0.94, 'Red bars push prediction toward 0 (No Churn) · Blue bars push toward 1 (Churn)',
              fontsize=9, ha='center', va='center', color='#555555')
    
    # Make the plot cleaner
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('#888888')
    plt.gca().spines['bottom'].set_color('#888888')
    
    plt.tight_layout(pad=1.2)
    
    # Display in Streamlit with interpretation text
    if st is not None:
        # Display the plot
        st.pyplot(plt.gcf())
        plt.close()
        
        # Add interpretation guide below the chart
        st.markdown("""<div style='background-color:#f8f9fa; padding:15px; border-radius:5px; margin-top:10px; font-size:0.9em'>
        <b>How to interpret this plot:</b><br>
        - <span style='color:#4a86e8'>Blue bars (positive values)</span> push the prediction <b>toward churn</b> (1)<br>
        - <span style='color:#e84a5f'>Red bars (negative values)</span> push the prediction <b>away from churn</b> (0)<br>
        - Longer bars have stronger impacts on the prediction<br>
        - The base value represents the model's average output across all samples
        </div>""", unsafe_allow_html=True)
    else:
        plt.close()
