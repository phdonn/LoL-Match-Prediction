# Predicting Match Outcomes in League of Legends: A Data-Driven Approach

# Introduction

This project analyzes professional League of Legends match data to understand the factors influencing match outcomes. The dataset includes detailed game statistics, and the analysis focuses on the relationship between gold difference at 25 minutes (`golddiffat25`) and match outcomes (`result`).

## Data Cleaning Steps and Exploratory Data Analysis

The dataset initially had 116,064 rows and 161 columns. The `golddiffat25` column, crucial for analysis, had 23,520 missing values, while `result` had none. Dropping rows with missing `golddiffat25` reduced the dataset to 92,544 rows. Both columns were converted to numeric types for compatibility, ensuring accurate and reliable analysis with complete data.

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
A histogram of `golddiffat25` shows a roughly normal distribution centered around zero, with most values between -5,000 and 5,000. This indicates that games are generally balanced at 25 minutes, with extreme gold differences being rare, reflecting the competitive nature of League of Legends.

### Bivariate Analysis  
Scatter and box plots reveal a positive relationship between `golddiffat25` and `result`. Winning teams (`result = 1`) typically have positive gold differences, while losing teams (`result = 0`) have negative or neutral ones, highlighting the importance of early-game gold leads in match outcomes.

### Aggregate Statistics

The dataset was grouped by match outcome (`result`), and the mean gold difference for each outcome was calculated. Teams that won their matches had an average gold lead of 1,511.96, while losing teams had an average gold deficit of -1,511.96. This significant difference reinforces the idea that gold advantage at 25 minutes is a key indicator of success in League of Legends matches.

### Grouped Table

```plaintext
   result  golddiffat25
0       0  -1511.962828
1       1   1511.962828
```

**Data Processing & Distribution**

*Missing Values*
- Dropped rows with missing `golddiffat25`
- No imputation to avoid bias in key metric

*Gold Difference Distribution*
- Normal distribution centered at 0
- Most values: -5,000 to +5,000
- Suggests balanced gameplay at 25min
- Small gold leads can be decisive

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


## Gold Difference Distribution
<iframe
  src="assets/gold_difference_distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

##### This graph provides a clear comparison of the distribution of gold differences at the 25-minute mark (`golddiffat25`) for winning (`result = 1`) and losing (`result = 0`) teams in League of Legends matches. The calculated correlation between golddiffat25 (gold difference at the 25-minute mark) and result (match outcome) is 0.478. This indicates a moderate positive correlation between the two variables. Here's an analysis of what the graph tells us:
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



**Model Evaluation**
F1-score is used over accuracy to handle potential class imbalance in win/loss predictions.

**Features**
Only includes data available at 25 minutes to maintain prediction validity and prevent data leakage.

**Significance**
Helps teams optimize early-game strategies by identifying key performance metrics that influence match outcomes.

### Baseline Model Description

Baseline Model
Using Logistic Regression via scikit-learn Pipeline for binary win/loss classification.
Performance

True Losses (TN): 10,417
True Wins (TP): 10,562
False Positives: 3,465
False Negatives: 3,320

Gold difference at 25 minutes proves to be a strong predictor, though misclassification errors suggest room for improvement.
### Baseline Model Performance:

|   True Labels |     0 |     1 |
|--------------:|------:|------:|
|             0 | 10417 |  3465 |
|             1 |  3320 | 10562 |





Grouped Statistics:
|   result |   golddiffat25 |
|---------:|---------------:|
|        0 |       -1511.96 |
|        1 |        1511.96 |

**Gold Difference Impact**
- Losing teams: -1,512 gold (avg)
- Winning teams: +1,512 gold (avg)

This clear separation confirms gold difference at 25 minutes as a strong predictor of match outcomes.

### Baseline Model Performance:


## Features in the Model
The model uses the following features:

1. **`golddiffat25`** (Quantitative):  
   This feature represents the gold difference at the 25-minute mark, providing a measure of the team’s economic advantage or disadvantage during the early to mid-game phase.

2. **`xpdiffat25`** (Quantitative):  
   This feature represents the experience difference at the 25-minute mark, giving insight into the team’s relative level advantage or deficit compared to their opponent.

Both features are quantitative and continuous. Since there were no ordinal or nominal features, encoding was not required. The features were scaled using **StandardScaler** within the pipeline to normalize their ranges and ensure improved logistic regression performance.

## Model Performance

**Model Performance on Test Set (30% of data)**
Model shows balanced performance across metrics and classes.

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


**Model Quality Assessment**
1. Features (`golddiffat25`, `xpdiffat25`) align with LoL game mechanics and are available at prediction time
2. Balanced metrics across wins/losses with 75.56% accuracy
3. Logistic regression provides clear feature importance interpretation


**Model Limitations**
- Misses complex feature interactions and nonlinear relationships
- Only uses two features (gold/XP diff)
- Logistic regression may oversimplify game dynamics

**Improvements**
- Add features: kills, deaths, objectives
- Test advanced models: Random Forests, Gradient Boosting
- Implement feature selection


### Final Model
## **Graph Analysis and Descriptions**

**Model Parameters & Performance**
- Best params: Random Forest with depth=10, 100 trees, sqrt features
- Accuracy: 75.4%
- Balanced precision/recall (~75%) across classes

**Feature Engineering**
1. K/D Ratio (kills/(deaths+1))
   - Measures teamfight efficiency
   - Captures survival impact on map pressure

2. Gold-XP Interaction (gold×XP diff)
   - Reflects compounding advantages
   - Combines item and level power spikes

**Feature Importance**
1. Gold diff: 0.4628
2. XP diff: 0.3777
3. Kills: 0.1595
4. Deaths: 0.0000

Original features remain strongest predictors, suggesting engineered features may be redundant.

Accuracy: 0.7539979829995678

Classification Report:
              precision    recall  f1-score   support

           0       0.75      0.76      0.76     13882
           1       0.76      0.75      0.75     13882

    accuracy                           0.75     27764
   macro avg       0.75      0.75      0.75     27764
weighted avg       0.75      0.75      0.75     27764

```
golddiffat25: 0.4628
xpdiffat25: 0.3777
killsat25: 0.1595
deathsat25: 0.0000
```

**What went wrong**
The original features likely dominate because:

1. Gold/XP differences are comprehensive metrics that inherently capture kill/death performance
   - Kills generate gold/XP
   - Deaths result in lost farm time
   - Original metrics already reflect teamfight outcomes

2. Gold-XP interaction may be redundant since:
   - These metrics naturally correlate
   - The model can learn this relationship from individual features
   - Linear combination might capture interaction better than multiplication

3. K/D ratio provides less signal because:
   - Raw numbers don't account for objective bounties
   - Kill value varies (shutdown gold, assists)
   - Deaths impact already reflected in resource differences

This suggests early-game resource advantages are more predictive than how teams acquired them.

### Modeling Algorithm and Hyperparameters:

The final model utilized a **Random Forest Classifier**, chosen for its ability to handle non-linear relationships and feature interactions while being resistant to overfitting through ensemble learning.

#### **Best Hyperparameters** (from GridSearchCV):
- `max_depth`: 10
- `min_samples_leaf`: 1
- `min_samples_split`: 2
- `n_estimators`: 100

These hyperparameters were selected using **GridSearchCV** with 3-fold cross-validation, exploring a comprehensive parameter space to find the optimal configuration.

### Comparison to Baseline Model:

1. **Performance Metrics**:
   - **Baseline Model Accuracy**: 76.56%
   - **Final Model Accuracy**: 75.39%
   - While the final model shows slightly lower accuracy, it provides more balanced predictions across classes.

2. **Feature Importance Analysis**:
   - The model heavily relies on the original features:
     - golddiffat25: 0.4628
     - xpdiffat25: 0.3777
     - killsat25: 0.1595
     - deathsat25: 0.0000
   - This suggests that the base features (gold and XP differences) are strong predictors of game outcomes at 25 minutes.

3. **Model Characteristics**:
   - The relatively shallow tree depth (max_depth=10) helps prevent overfitting
   - Low minimum samples parameters allow the model to capture fine-grained patterns
   - 100 trees provide a good balance between model complexity and performance

**Final Assessment**
Though accuracy remained similar, final model offers:
- More balanced predictions across different scenarios  
- Better feature interpretability via gameplay metrics
- Tuned hyperparameters balancing complexity/generalization

Makes it more reliable for practical game prediction despite minimal accuracy gains.

## **Graph 1: Baseline Feature Importance**
   <iframe
    src="assets/accuracy_comparison_2.html"
    width="800"
    height="600"
    frameborder="0"
    ></iframe>

- **Description**: Both models performed similarly (75.6% vs 75.4% accuracy), suggesting added complexity provided no meaningful improvement over the baseline's key features.
---

## **Graph 2: Final Feature Importance**

<iframe
    src="assets/feature_importance_final_2.html"
    width="800"
    height="600"
    frameborder="0"
    ></iframe>


**Feature Importance**
- Gold diff (25min): 0.46 
- XP diff: 0.378
- Kills: 0.159
- Deaths: 0.000

Resource-based metrics (gold/XP) strongly outperform combat stats. Kills show moderate impact while deaths have no predictive value, suggesting advantage-gaining matters more than avoiding losses.
---

## **Graph 4: Comparison of Metrics by Class**

  <iframe
  src="assets/model_metrics_@.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
- **Description**: 
**Model Metrics by Class**
- Losses: All metrics ~0.76
- Wins: Precision 0.76, recall/F1 0.75
- Balanced performance across classes (baseline 0.75)

Model shows no bias between predicting wins vs losses.
---

## **Graph 5: Confusion Matrices**
<iframe
  src="assets/confusion_matrix_2.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

- **Description**:Confusion Matrix
The matrix shows prediction outcomes with:


True Negatives (Losses correctly predicted): 10,550
False Positives: 3,332
False Negatives: 3,472
True Positives (Wins correctly predicted): 10,410
This balanced distribution suggests the model performs similarly well for both winning and losing predictions, without significant bias toward either outcome.

---

## **Graph 6: ROC Curve Comparison**

<iframe
  src="assets/roc_curve_2.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
- **Description**: ROC curve shows strong prediction capability (AUC=0.830), well above random (0.5). Sharp initial rise indicates high confidence in strongest predictions.

---
## **Feature Engineering and Selection**
- **Added Features**:
  1. **`kills_deaths_ratio`**: While intended to capture team fight efficiency, this feature ultimately did not contribute significantly to the model's performance, as shown by the feature importance analysis.
  2. **`gold_xp_interaction`**: Similarly, this engineered feature did not appear as significant in the final model, suggesting that the raw gold and XP differences were more informative.
- **Feature Importance Results**: The analysis showed that the original features were most predictive:
  - golddiffat25: 0.46 importance
  - xpdiffat25: 0.378 importance
  - killsat25: 0.159 importance
  - deathsat25: 0.000 importance

## **Modeling Algorithm and Hyperparameter Tuning**
- **Algorithm**: Random Forest Classifier, chosen for its ability to handle non-linear relationships and feature interactions.
- **Best Hyperparameters** (from GridSearchCV):
  - `max_depth = 10`
  - `min_samples_leaf = 1`
  - `min_samples_split = 2`
  - `n_estimators = 100`
- **Tuning Method**: GridSearchCV with 3-fold cross-validation.

### **Performance Comparison**
- **Baseline Model**:
  - Accuracy: 75.56%
  - Equal performance across classes
- **Final Model**:
  - Accuracy: 75.39%
  - Confusion Matrix shows balanced prediction:
    - True Negatives (Losses): 10,550
    - False Positives: 3,332
    - False Negatives: 3,472
    - True Positives (Wins): 10,410
  - ROC-AUC: 0.830, indicating strong discriminative ability

## Conclusion

This project explored predicting League of Legends match outcomes using data at the 25-minute mark. While our feature engineering attempts (kills/deaths ratio and gold/XP interaction) were theoretically sound, the empirical results showed that the original features were most predictive.

### Key Findings:
1. **Resource Advantages Dominate**:
   - Gold difference (0.46) and XP difference (0.378) were the strongest predictors
   - Kill count had moderate importance (0.159)
   - Death count showed no significant predictive power

2. **Model Performance**:
   - The final model achieved 75.39% accuracy
   - Strong ROC-AUC score of 0.830
   - Balanced performance across winning and losing predictions

3. **Feature Engineering Impact**:
   - Simpler features outperformed engineered ones
   - Original resource-based metrics (gold, XP) captured most of the predictive signal

This analysis suggests that in professional League of Legends, resource advantages at 25 minutes are more reliable predictors of game outcomes than combat statistics or engineered feature combinations. The high ROC-AUC score indicates that the model makes reliable probabilistic predictions, even though the accuracy remained similar to the baseline model.