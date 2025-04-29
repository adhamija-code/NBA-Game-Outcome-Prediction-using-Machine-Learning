# NBA-Game-Outcome-Prediction-using-Machine-Learning
This project is the final submission for CSE 802: Pattern Recognition, focused on building and evaluating machine learning models to predict the outcome of NBA games (home team win or loss) using historical and contextual game data.

🔍 Objective
Develop a complete machine learning pipeline that:

Preprocesses and enriches raw NBA game data

Applies various normalization and dimensionality reduction strategies

Trains and evaluates multiple classifiers, including a custom Bayesian Classifier

⚙️ Techniques Used
Normalization: None, Min-Max Scaling, Z-Score Standardization

Dimensionality Reduction: PCA, SFFS (Sequential Forward Floating Selection)

Classification Models:

Logistic Regression

Random Forest

Support Vector Machines (SVM)

Bayesian Classifiers (Naive, Multivariate, Nonparametric)

📊 Key Results
Best Performing Setup:
Z-Score + SFFS + SVM → Accuracy: 99.89% ± 0.02%

SFFS outperformed PCA in all scenarios

Z-Score Normalization yielded the highest consistency and model stability

📁 Project Structure
├── main.py / main.ipynb           # Full model training and evaluation pipeline
├── preprocess.py                  # Data loading, merging, cleaning, and transformations
├── bayesian_classifier_custom.py # Implementation of multivariate, naive, and KDE classifiers
├── custom_pca_sffs.py             # Custom PCA and SFFS dimensionality reduction
├── evaluate.py                    # Model evaluation utilities (confusion matrix, metrics)
├── final_graph.ipynb             # Aggregated result graphs and comparisons
├── games.csv, teams.csv, etc.    # Raw and enriched datasets
📚 Report
The full project report (PDF) can be found here, detailing the methodology, experimental design, accuracy tables, graphs, and analysis.

📌 Future Work
Integrating real-time player stats and in-game context

Addressing class imbalance using SMOTE

Exploring deep learning architectures (e.g., LSTM)
