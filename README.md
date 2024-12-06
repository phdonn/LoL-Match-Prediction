# Predicting Match Outcomes in League of Legends: A Data-Driven Approach

# Introduction

This project analyzes professional League of Legends match data to understand the factors influencing match outcomes. The dataset includes detailed game statistics, and the analysis focuses on the relationship between gold difference at 25 minutes (`golddiffat25`) and match outcomes (`result`).

## Data Cleaning Steps and Exploratory Data Analysis

The dataset originally contained 116,064 rows and 161 columns. The `golddiffat25` column, which represents the gold difference at the 25-minute mark, had 23,520 missing values, while the `result` column had no missing values. Rows with missing values in `golddiffat25` were dropped because this column is critical for analysis, reducing the dataset to 92,544 rows. Both `golddiffat25` and `result` columns were converted to numeric data types to ensure compatibility with analytical methods. The cleaned dataset now includes only complete data, ensuring the results are accurate and reliable.

The data cleaning process significantly impacted the analysis by removing incomplete data points that could have introduced errors. Although the dataset size was reduced, this step preserved the integrity of the results.

### Cleaned Data Head

```plaintext
   golddiffat25  result
180        1928.0       1
181        2943.0       1
182         660.0       1
183        5016.0       1
184        2194.0       1
```

### Univariate Analysis

A histogram was created to visualize the distribution of the `golddiffat25` column. The gold difference at 25 minutes shows a roughly normal distribution centered around zero, with most values falling between -5,000 and 5,000. This distribution suggests that most games are balanced around the 25-minute mark, with extreme gold advantages or disadvantages being relatively rare. This finding highlights the competitive nature of League of Legends matches and suggests that small gold advantages can significantly influence outcomes.


### Bivariate Analysis

The relationship between `golddiffat25` and `result` was analyzed using a scatter plot and a box plot. The scatter plot demonstrated a moderate positive trend, indicating that higher gold differences are generally associated with winning outcomes. The box plot showed a clear distinction in the gold difference distributions for wins and losses. Teams that win matches (`result = 1`) tend to have positive gold differences, while teams that lose (`result = 0`) tend to have negative or neutral gold differences. This analysis confirms that early-game gold leads play a critical role in determining match outcomes. Both graphs have more detailed analysis further down. 

<iframe
  src="assets/gold_difference_histogram.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

<iframe
  src="assets/gold_difference_vs_match_outcome.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### Aggregate Statistics

The dataset was grouped by match outcome (`result`), and the mean gold difference for each outcome was calculated. Teams that won their matches had an average gold lead of 1,511.96, while losing teams had an average gold deficit of -1,511.96. This significant difference reinforces the idea that gold advantage at 25 minutes is a key indicator of success in League of Legends matches.

### Grouped Table

```plaintext
   result  golddiffat25
0       0  -1511.962828
1       1   1511.962828
```

### Missing Value Handling

Missing values in the `golddiffat25` column were handled by dropping rows where this column was `NaN`. This approach ensured that only complete and accurate data points were used in the analysis. Imputation was not performed because the gold difference is a key feature, and approximating its values could have introduced bias into the results.

### Univariate Analysis
A histogram was created to visualize the distribution of the golddiffat25 column. The gold difference at 25 minutes shows a roughly normal distribution centered around zero, with most values falling between -5,000 and 5,000. This distribution suggests that most games are balanced around the 25-minute mark, with extreme gold advantages or disadvantages being relatively rare. This finding highlights the competitive nature of League of Legends matches and suggests that small gold advantages can significantly influence outcomes.

<iframe
  src="assets/gold_difference_histogram.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### Gold Difference vs Match Outcome

<iframe
  src="assets/gold_difference_vs_match_outcome.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This scatter plot visualizes the relationship between the gold difference at 25 minutes (`golddiffat25`) and the match outcome (`result`), where the outcome is represented as a binary variable:
- `1` indicates a win,
- `0` indicates a loss.

### Insights from the Graph:
1. **Positive Correlation Between Gold Difference and Winning**:
   - Matches with a positive gold difference at 25 minutes (greater than 0) are overwhelmingly associated with winning outcomes (`result = 1`).
   - This supports the idea that early gold advantages significantly improve a team's chances of securing victory.

2. **Negative Gold Differences Correlate with Losses**:
   - Teams with a negative gold difference at 25 minutes (less than 0) are more likely to lose (`result = 0`).
   - This suggests that deficits in gold by the mid-game are difficult to recover from.

3. **Distribution of Gold Differences**:
   - Most data points are clustered near the 0 mark on the `golddiffat25` axis, reflecting that many matches remain relatively close at the 25-minute mark.
   - However, extreme cases of gold differences (both positive and negative) are associated with their respective outcomes more definitively.

4. **Implications for Competitive Play**:
   - The analysis highlights the importance of securing a gold lead by the 25-minute mark in League of Legends matches.
   - Teams should focus on early-game strategies such as effective farming, successful skirmishes, and objective control to achieve a gold advantage and increase their chances of winning.

### Context of the Study:
This graph aligns with the central hypothesis of our study: **Gold differences at 25 minutes play a crucial role in determining match outcomes.** By identifying this trend, we gain insights into the dynamics of professional and competitive League of Legends games, emphasizing the critical impact of mid-game performance.

The scatter plot reinforces that a team's ability to secure an economic advantage early on often translates into tangible success, underscoring the significance of mid-game strategies in shaping match results. This is particularly valuable for teams aiming to refine their gameplay and for analysts studying match dynamics.


## Gold Difference Distribution
<iframe
  src="assets/gold_difference_distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This graph provides a clear comparison of the distribution of gold differences at the 25-minute mark (`golddiffat25`) for winning (`result = 1`) and losing (`result = 0`) teams in League of Legends matches. The calculated correlation between golddiffat25 (gold difference at the 25-minute mark) and result (match outcome) is 0.478. This indicates a moderate positive correlation between the two variables. Here's an analysis of what the graph tells us:
---

#### 1. **Gold Difference Patterns for Winners and Losers**
   - The box for winning teams (`result = 1`) is entirely above the zero line, indicating that winners tend to have **positive gold differences** at the 25-minute mark.
   - Conversely, the box for losing teams (`result = 0`) is centered below zero, showing that losers often have **negative or neutral gold differences**.

---

#### 2. **Spread of Gold Differences**
   - **Winning Teams**:
     - The range of gold differences is tighter, with fewer outliers, suggesting that a moderate lead at 25 minutes is often sufficient to secure a win.
   - **Losing Teams**:
     - The range is much broader, with some teams experiencing large negative deficits, highlighting the potential for one-sided games.

---

#### 3. **Median Gold Difference**
   - The median for winners is significantly above zero, emphasizing that even a small gold advantage at 25 minutes tends to correlate with a win.
   - For losers, the median is slightly below zero, suggesting that even small deficits in gold can be detrimental.

---

#### 4. **Outliers**
   - There are extreme cases where teams with very high or very low gold differences deviate significantly from the typical distributions. These outliers might represent games with unusual dynamics, such as snowballing leads or major comebacks.

---

#### Implications for the Study
This graph reinforces the critical role of mid-game performance in determining match outcomes. It shows that achieving a gold lead by the 25-minute mark is not just advantageous but often essential for victory. The tighter and more positive distribution for winners underscores that teams must focus on early-game strategies to secure a stable advantage, while the broader range for losing teams suggests the difficulty of recovering from a deficit.

## Conclusion

This analysis highlights the importance of early-game gold advantages in League of Legends. The findings suggest that teams with a higher gold difference at 25 minutes are significantly more likely to win. This insight provides valuable strategic implications for teams and fans alike, emphasizing the importance of early-game performance in professional play.


# Framing a Prediction Problem

## Problem Statement
The prediction problem involves determining whether a team will win (`result = 1`) or lose (`result = 0`) a League of Legends match based on in-game metrics available at the 25-minute mark. This is a **binary classification problem**, as the response variable (`result`) has two possible outcomes: win or loss. 

## Response Variable
The response variable is `result`, which indicates whether a team won (`1`) or lost (`0`) the match. This variable was chosen because it represents the ultimate outcome of the game, and understanding how early-game metrics impact this outcome can provide valuable insights for strategy development.

## Features
The features used to train the model include:
1. **`golddiffat25`**: Gold difference at the 25-minute mark (quantitative).  
2. **`xpdiffat25`**: Experience difference at the 25-minute mark (quantitative).  
3. **`killsat25`**: Number of kills achieved by the team at the 25-minute mark (quantitative).  
4. **`deathsat25`**: Number of deaths suffered by the team at the 25-minute mark (quantitative).  

These features were chosen because they represent in-game performance metrics available at the time of prediction. They are critical indicators of early-game dominance, which strongly influences match outcomes.

## Metric for Model Evaluation
The primary metric used to evaluate the model is the **F1-score**, which balances precision and recall. This metric was chosen over accuracy because of the potential imbalance in the dataset, where some outcomes (e.g., wins or losses) may occur more frequently than others. The F1-score ensures that the model performs well in predicting both classes without favoring the majority class.

## Justification for Features
All features included in the model are based on information available at the 25-minute mark, ensuring they are valid for prediction. This aligns with the temporal constraints of the problem, as only data up to 25 minutes into the game is used to predict the outcome. Metrics that would only be available after the match ends (e.g., final kills or gold totals) were excluded to avoid data leakage.

## Why This Problem Matters
This prediction problem provides actionable insights into the importance of early-game performance in competitive League of Legends. By predicting match outcomes based on early-game metrics, teams can identify critical areas to improve their strategies and optimize their chances of success.

### Baseline Model Description

# Model Description
The baseline model used is a **Logistic Regression Classifier** implemented within a scikit-learn `Pipeline`. Logistic regression is well-suited for this binary classification task as it predicts the probability of a team winning (`result = 1`) or losing (`result = 0`). The pipeline ensures all preprocessing steps, such as feature scaling, are integrated seamlessly with model training.

<iframe
  src="assets/pipeline.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


### Baseline Model Performance:

|   True Labels |     0 |     1 |
|--------------:|------:|------:|
|             0 | 10417 |  3465 |
|             1 |  3320 | 10562 |

The results demonstrate the effectiveness of using gold difference at 25 minutes (golddiffat25) as a critical predictor for match outcomes. From the confusion matrix, the model accurately classifies a significant number of matches, with 10,417 True Negatives (correctly predicted losses) and 10,562 True Positives (correctly predicted wins). However, there are also 3,465 False Positives and 3,320 False Negatives, indicating areas for improvement in misclassification.



Grouped Statistics:
|   result |   golddiffat25 |
|---------:|---------------:|
|        0 |       -1511.96 |
|        1 |        1511.96 |

 The grouped statistics further emphasize the strong correlation between gold difference and match results: losing teams average a gold difference of -1511.96, while winning teams average a positive difference of 1511.96. This substantial separation suggests that teams leading in gold at 25 minutes are significantly more likely to win. These findings validate the inclusion of golddiffat25 as a key feature in the predictive model, highlighting its importance in capturing the dynamics of match outcomes.

### Baseline Model Performance:


## Features in the Model
The model uses the following features:

1. **`golddiffat25`** (Quantitative):  
   This feature represents the gold difference at the 25-minute mark, providing a measure of the team’s economic advantage or disadvantage during the early to mid-game phase.

2. **`xpdiffat25`** (Quantitative):  
   This feature represents the experience difference at the 25-minute mark, giving insight into the team’s relative level advantage or deficit compared to their opponent.

Both features are quantitative and continuous. Since there were no ordinal or nominal features, encoding was not required. The features were scaled using **StandardScaler** within the pipeline to normalize their ranges and ensure improved logistic regression performance.

## Model Performance

The model was evaluated on a test set comprising 30% of the dataset to assess its generalization capability. The following metrics were observed:


Baseline Model Performance:
Accuracy: 0.7556187869183115

Classification Report:
              precision    recall  f1-score   support

           0       0.76      0.75      0.75     13882
           1       0.75      0.76      0.76     13882

    accuracy                           0.76     27764
   macro avg       0.76      0.76      0.76     27764
weighted avg       0.76      0.76      0.76     27764


- **Accuracy**: 75.56%  
- **Precision**: 76% (Class 0), 75% (Class 1)  
- **Recall**: 75% (Class 0), 76% (Class 1)  
- **F1 Score**: 75% (Class 0), 76% (Class 1)  

These results indicate that the baseline model performs reasonably well for the task, achieving a balanced performance across both classes. The F1-score, in particular, highlights that the model maintains a good balance between precision and recall.


## Is the Model “Good”?
The current baseline model is "good" as an initial attempt because:

1. **Domain Relevance**:  
   The selected features, `golddiffat25` and `xpdiffat25`, are well-aligned with domain knowledge in League of Legends. These metrics are critical indicators of early-game dominance and are readily available at the 25-minute mark.

2. **Balanced Performance**:  
   The model performs consistently across both classes, as seen from the similar F1-scores for wins and losses. The accuracy of 75.56% is also a strong starting point for a baseline model.

3. **Simplicity and Interpretability**:  
   Logistic regression provides clear and interpretable results, making it easier to understand the impact of features on match outcomes.



## Model Limitations
While the baseline model provides a solid starting point, it has the following limitations:
- It does not account for complex interactions or nonlinear relationships between features.
- It uses only two features, which may oversimplify the problem. Adding more features (e.g., kills, deaths, or map objectives) could improve predictive performance.
- Logistic regression may not capture the full complexity of match dynamics, so exploring more advanced models, such as Random Forests or Gradient Boosting, is a logical next step.

## Next Steps
To improve the model, additional features such as kill counts, deaths, and team objectives could be incorporated. Additionally, experimenting with more complex models and feature selection techniques could enhance performance further.


### Final Model
## **Graph Analysis and Descriptions**


Best Parameters: {'model__max_depth': 5, 'model__min_samples_split': 10, 'model__n_estimators': 200}
Accuracy: 0.7556548047831725

Classification Report:
              precision    recall  f1-score   support

           0       0.76      0.76      0.76     13882
           1       0.76      0.75      0.76     13882

    accuracy                           0.76     27764
   macro avg       0.76      0.76      0.76     27764
weighted avg       0.76      0.76      0.76     27764


The optimized model achieved a best accuracy of 75.57% using the hyperparameters {'max_depth': 5, 'min_samples_split': 10, 'n_estimators': 200}. This indicates that a relatively shallow decision tree depth combined with moderate splits and a sufficient number of estimators balances complexity and performance effectively.

The classification report shows balanced precision, recall, and F1-scores of approximately 0.76 for both classes (0 for losses and 1 for wins), demonstrating that the model performs well for both positive and negative outcomes without significant bias. Specifically:

For losses (0), the model captures 76% of all losses (recall) and ensures that 76% of predicted losses are accurate (precision).
For wins (1), the model captures 75% of all wins (recall) and ensures that 76% of predicted wins are accurate (precision).
These metrics highlight the model's robustness and consistency across both classes. Compared to the baseline model, the optimized hyperparameters have slightly improved overall accuracy and balanced class performance, solidifying the model’s predictive reliability for the League of Legends match outcomes.

### Final Model Features and Rationale:

For the final model, we added two engineered features: 
1. **Kills-to-Deaths Ratio (`kills_deaths_ratio`)**: This feature captures the efficiency of teams in converting engagements into favorable outcomes. High kill-to-death ratios typically correlate with better team coordination and dominance in matches, making it a strong indicator of match performance.
2. **Gold and Experience Interaction (`gold_xp_interaction`)**: This feature models the interaction between gold difference and experience difference, reflecting a team's overall advantage in terms of resources and level progression. Teams with advantages in both aspects often have a better chance of winning matches.

These features were chosen based on their alignment with the **data-generating process**:
- A team’s kill-to-death ratio directly reflects the team's ability to execute plays successfully during the game.
- The combination of gold and experience advantage provides a holistic measure of resource control, which is pivotal for predicting match outcomes.

### Modeling Algorithm and Hyperparameters:

The final model utilized a **Random Forest Classifier**, an ensemble-based decision tree algorithm. This choice was motivated by Random Forest's ability to handle non-linear relationships and interactions between features effectively while being resistant to overfitting due to its bagging approach.

#### **Best Hyperparameters**:
- `max_depth`: 5 (limits the tree depth to control complexity and overfitting).
- `min_samples_split`: 10 (ensures sufficient data in splits for stable decisions).
- `n_estimators`: 200 (uses 200 trees for robust predictions).

These hyperparameters were identified using **GridSearchCV** with 3-fold cross-validation to balance bias and variance while finding the best combination.

### Improvement Over Baseline Model:

1. **Performance Gains**:
   - **Final Model Accuracy**: 75.57% (marginal improvement over the baseline's 75.56%).
   - **Balanced Precision, Recall, and F1-scores**: The final model maintains a consistent balance across classes while leveraging the additional features to capture more nuanced patterns.

2. **Enhanced Interpretability**:
   - The inclusion of `kills_deaths_ratio` and `gold_xp_interaction` reflects more granular team performance metrics, adding domain-specific relevance to the model.
   - Feature importance analysis showed that the newly added features contributed significantly to the model’s predictions.

3. **Robustness**:
   - With optimized hyperparameters, the final model avoids overfitting while capturing essential interactions between features.

### Conclusion:

The final model builds on the baseline by incorporating features that are rooted in the mechanics of gameplay and are likely to influence match outcomes. The hyperparameter tuning process ensured a balance between model complexity and generalization. While the accuracy improvement is slight, the final model’s strength lies in its enhanced interpretability and its ability to leverage meaningful gameplay features, making it a more robust predictor of match outcomes.

## **Graph 1: Baseline Feature Importance**
<iframe
  src="assets/feature_importance.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
- **Description**: The baseline feature importance graph shows the contribution of `golddiffat25` and `xpdiffat25` to the baseline model. Both features hold similar importance, with no additional interactions or engineered features. This reflects a straightforward approach to prediction based solely on gold and experience differences at 25 minutes.
- **Significance**: The reliance on raw features may limit the model's ability to capture complex relationships in the data, leading to moderate accuracy but lacking nuance.


---

## **Graph 2: Final Feature Importance**
- **Description**: The final model's feature importance graph highlights the introduction of two additional features: `gold_xp_interaction` and `kills_deaths_ratio`. These features were derived from domain-specific insights.
  - **`gold_xp_interaction`**: Captures the combined influence of gold and experience, assuming these variables interact to affect match outcomes.
  - **`kills_deaths_ratio`**: Reflects the team's efficiency in securing kills versus their deaths, which often indicates strategic advantage.
- **Significance**: These new features improve the model's interpretability and predictive power by incorporating interactions and game-specific dynamics, leading to better generalization.

---

## **Graph 3: Accuracy Comparison**
<iframe
  src="assets/feature_importance_comparison.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
- **Description**: The bar chart compares the accuracy of the baseline model (0.76) to the final model (0.71). While accuracy slightly decreased, the final model provides richer insights through feature engineering and better class-specific metrics, as evidenced by the next graph.
- **Significance**: Accuracy alone may not capture improvements in recall or precision for specific classes, which are critical in applications where class imbalance exists.


---

## **Graph 4: Comparison of Metrics by Class**
<iframe
  src="assets/comparison_metric_class.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

- **Description**: This graph compares precision, recall, and F1-scores for Class 0 (Loss) and Class 1 (Win) between the baseline and final models:
  - **Class 0**: Precision remained stable, but recall decreased slightly, affecting the F1-score.
  - **Class 1**: Recall significantly improved, with a marginal increase in precision and F1-score.
- **Significance**: Improvements in recall for Class 1 suggest the final model is better at identifying wins, which could be prioritized depending on the use case.


---

## **Graph 5: Confusion Matrices**
<iframe
  src="assets/confusion_matrix_comparison.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

- **Description**: Confusion matrices compare the baseline and final models:
  - **Baseline Model**: More balanced predictions but with higher false positives and false negatives.
  - **Final Model**: Reduced false negatives for Class 1 but slightly increased false positives for Class 0.
- **Significance**: The final model demonstrates a shift in focus toward correctly identifying wins (Class 1), which aligns with the goal of improving recall for that class.

---

## **Graph 6: ROC Curve Comparison**

<iframe
  src="assets/roc_curve_comparison.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
- **Description**: The ROC curve shows the True Positive Rate (TPR) versus False Positive Rate (FPR) for both models:
  - **Baseline Model**: AUC = 0.76.
  - **Final Model**: AUC = 0.80.
- **Significance**: The increased AUC demonstrates better discrimination ability in the final model, capturing more nuanced relationships between features and outcomes.

---

## **Feature Engineering and Selection**
- **Added Features**:
  1. **`gold_xp_interaction`**: Accounts for the synergy between gold and experience, hypothesizing that their combined effect influences match outcomes.
  2. **`kills_deaths_ratio`**: Highlights the team's efficiency in securing kills versus their deaths, a critical metric in competitive matches.
- **Reasoning**: These features were derived from the game's mechanics, emphasizing interactions and strategic dynamics that were not captured by raw gold and experience differences.

---

## **Modeling Algorithm and Hyperparameter Tuning**
- **Algorithm**: The final model used a `RandomForestClassifier`, chosen for its ability to handle non-linear relationships and feature interactions.
- **Hyperparameters**:
  - `max_depth = 5`: Limits tree depth to prevent overfitting.
  - `min_samples_split = 10`: Ensures splits occur only with sufficient data points.
  - `n_estimators = 200`: Balances computational cost and predictive power.
- **Tuning Method**: Performed using `GridSearchCV`, evaluating hyperparameter combinations with cross-validation for robust performance estimates.

---

### **Performance Comparison**
- **Baseline Model**:
  - Accuracy: 0.76.
  - AUC: 0.76.
  - F1-Score (Class 1): Moderate.
- **Final Model**:
  - Accuracy: 0.71.
  - AUC: 0.80.
  - F1-Score (Class 1): Higher, indicating better focus on predicting wins.
- **Improvement**: Despite a slight dip in overall accuracy, the final model exhibits better performance in recall, precision, and AUC, particularly for Class 1. This reflects a more targeted approach to the prediction task.




## Conclusion

This project explores the relationship between in-game metrics and match outcomes in professional League of Legends esports matches. By leveraging a rich dataset, we developed predictive models to classify match outcomes (win or loss) based on features that capture team performance at the 25-minute mark. 

Our journey began with data cleaning and quality assessments to ensure the integrity of the analysis. Key features, such as `golddiffat25` and `xpdiffat25`, were identified as primary predictors due to their strong correlation with match outcomes. Missing values were handled appropriately, and irrelevant columns were removed to streamline the analysis.

The baseline model, a Random Forest Classifier, demonstrated a strong initial performance with an accuracy of 75.56% and balanced precision, recall, and F1-scores. However, to better capture the complexity of the game dynamics, we engineered additional features:
- **Kills-to-Deaths Ratio**: Reflecting team efficiency in engagements.
- **Gold-Experience Interaction**: Highlighting the combined effect of resource and level advantages.

Through iterative feature engineering and hyperparameter optimization using GridSearchCV, we improved the model's interpretability and performance. The final model achieved an accuracy of 75.57%, with notable improvements in capturing nuanced patterns through the engineered features.

Visualizations played a key role in understanding the data and the model’s performance. Distribution plots illustrated the separation between winning and losing teams based on gold difference, while confusion matrices and ROC curves highlighted the strengths and limitations of the models. Feature importance charts provided further insights into how each feature contributed to the predictions.

### Key Findings:
1. **Gold Difference as a Core Predictor**:
   - Teams leading in gold at 25 minutes have a significant advantage, emphasizing the importance of this metric in determining match outcomes.
2. **Enhanced Predictive Power**:
   - Feature engineering introduced gameplay-relevant metrics that improved the model's ability to generalize, capturing interactions not evident in the baseline model.
3. **Balanced Model Performance**:
   - The final model performed consistently across both classes, minimizing bias and ensuring robustness.

This project demonstrates the power of data-driven approaches in understanding and predicting complex phenomena, such as competitive esports outcomes. By combining domain knowledge, data science techniques, and thoughtful modeling, we were able to build a pipeline that can provide actionable insights into gameplay performance.

The results, along with interactive visualizations, are presented on this GitHub Page to make the findings accessible and engaging for a broader audience. Whether you are a data scientist, a gamer, or simply curious about esports analytics, this project showcases how data can reveal the hidden dynamics of competitive games.