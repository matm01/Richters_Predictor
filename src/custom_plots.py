import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def plot_damage_by_feature(df, feature):
    """
    Create a horizontal stacked bar plot of the damage grade by the selected feature.

    Args:
    df: DataFrame
    feature: str, the feature to be plotted
    """
    
    df_plot = df.copy()

    # Check if the feature is in the DataFrame
    if feature not in df_plot.columns:
        raise ValueError(f"The feature {feature} is not in the DataFrame.")    

    
    # Convert the feature to string if it is not
    if df_plot[feature].dtype != "object":
        # First check if the feature has more than 10 unique values
        if df_plot[feature].nunique() > 10:
            raise ValueError(f"The feature {feature} has more than 10 unique values.")
        # If not, convert it to string
        else:
            df_plot[feature] = df_plot[feature].astype(str)
            
    # Get unique categories in feature and damage_grade
    feature_categories = df_plot[feature].unique()
    damage_grade_categories = df_plot["damage_grade"].unique()

    # Generate all combinations of categories
    combinations = list(itertools.product(feature_categories, damage_grade_categories))

    # Create a dataframe with pre-filled rows
    summary_df = pd.DataFrame(combinations, columns=["feature", "damage_grade"])
    summary_df["proportion"] = 0.
    summary_df = summary_df.sort_values(by=["feature", "damage_grade"])

    # Fill in the proportions of damage grade within each category of feature
    for category in feature_categories:
        for damage_grade in damage_grade_categories:
            proportion = df_plot[(df_plot[feature] == category) & (df_plot["damage_grade"] == damage_grade)].shape[0] / df_plot[df_plot[feature] == category].shape[0]
            summary_df.loc[(summary_df["feature"] == category) & (summary_df["damage_grade"] == damage_grade), "proportion"] = proportion


    # Older approach that does not work with missing category combinations:
    # summary = df_plot.groupby(feature)["damage_grade"].value_counts(normalize=True)
    # summary_df = pd.DataFrame(summary).reset_index().sort_values([feature, "damage_grade"])
    

    # For plotting, extract the categories and the proportions of each damage grade
    categories = summary_df["feature"].unique().tolist()
    categories = np.sort(categories)
    segment_1 = summary_df[summary_df["damage_grade"] == 1]["proportion"].values
    segment_2 = summary_df[summary_df["damage_grade"] == 2]["proportion"].values
    segment_3 = summary_df[summary_df["damage_grade"] == 3]["proportion"].values

    # Create a horizontal stacked bar plot
    plt.barh(categories, segment_1, color='gold', label='Damage grade 1')
    plt.barh(categories, segment_2, left=segment_1, color='darkorange', label='Damage grade 2')
    plt.barh(categories, segment_3, left=np.add(segment_1, segment_2), color='crimson', label='Damage grade 3')

    # Adding data labels
    for i, (cat, seg1, seg2, seg3) in enumerate(zip(categories, segment_1, segment_2, segment_3)):
        plt.text(seg1/2, i, f'{seg1:.2f}', ha='center', va='center', color='white', bbox=dict(facecolor='black', edgecolor='none', pad=1))
        plt.text(seg1 + seg2/2, i, f'{seg2:.2f}', ha='center', va='center', color='white', bbox=dict(facecolor='black', edgecolor='none', pad=1))
        plt.text(seg1 + seg2 + seg3/2, i, f'{seg3:.2f}', ha='center', va='center', color='white', bbox=dict(facecolor='black', edgecolor='none', pad=1))

    # Adding labels and title
    plt.xlabel('Relative frequency of damage grade')
    plt.ylabel(feature)
    plt.title(f'Damage grade by {feature}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


    # Show the plot
    plt.show()

def make_confusion_matrix(y_valid, preds):
    conf_mat = confusion_matrix(y_valid, preds)

    conf_mat_df = pd.DataFrame(conf_mat, index=['Actual 0', 'Actual 1', 'Actual 2'], columns=['Predicted 0', 'Predicted 1', 'Predicted 2'])

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat_df, annot=True, fmt='d', cmap='Reds')
    plt.title('Confusion Matrix')
    plt.show()