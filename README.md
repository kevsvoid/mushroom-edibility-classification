# Mushroom Edibility Classification

## Project Overview
This project applies a Gradient Boosting Classifier to the UCI Mushroom Dataset to classify mushrooms as either edible or poisonous. The dataset consists of categorical morphological features such as odor, cap color, gill color, and spore print color.

The objective is to evaluate how effectively a tree-based ensemble model can learn patterns from structured categorical data and identify the most important features contributing to classification.

## Key Findings
- **Perfect Classification:** The model achieved 100% accuracy and AUC-ROC on the test set, indicating complete separation between edible and poisonous classes.
- **Dominant Feature:** Odor was identified as the most influential feature, showing near-deterministic relationships with the target class.
- **High Separability:** The dataset exhibits strong feature-class dependencies, making it highly suitable for tree-based models such as Gradient Boosting.

## Data Pipeline
The analysis is driven by a Python script (`mushroom_classification.py`) that performs:
1. **Data Loading:** Importing the UCI Mushroom dataset from CSV format.
2. **Preprocessing:** Standardizing column names and encoding categorical variables using Label Encoding.
3. **Train-Test Split:** Performing an 80/20 stratified split to preserve class distribution.
4. **Model Training:** Training a Gradient Boosting Classifier with default parameters.
5. **Evaluation:** Computing accuracy, confusion matrix, and ROC curve.
6. **Visualization:** Generating plots for dataset distribution, feature analysis, model performance, and feature importance.

## Repository Structure
- `data/`
    - `mushrooms.csv`: The original dataset from the UCI repository.
- `outputs/`
    - `fig1_dataset.png`: Dataset overview (class distribution + correlation heatmap)
    - `fig2_features.png`: Feature distributions (odor and cap color)
    - `fig3_confusion.png`: Confusion matrix
    - `fig4_roc.png`: ROC curve
    - `fig5_importance.png`: Feature importance visualization
- `mushroom_classification.py`: Main script for preprocessing, training, evaluation, and visualization.
- `README.md`: Project documentation.

## How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/kevsvoid/mushroom-edibility-classification.git(https://github.com/kevsvoid/mushroom-edibility-classification.git)
2. **Install dependencies:**
 Ensure you have `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, installed.
3. **Run the script:**
   ```terminal
   python mushroom_classification.py
3. **View outputs:**
   All generated figures will be saved in the `outputs/` folder.

---

### References
**UCI Machine Learning Repository. 1987. Mushroom Dataset.** University of California, Irvine.