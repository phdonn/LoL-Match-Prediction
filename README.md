```markdown
# League of Legends Match Analysis

This project analyzes professional League of Legends match data to understand the factors influencing match outcomes. The dataset includes detailed game statistics, and the analysis focuses on the relationship between gold difference at 25 minutes (`golddiffat25`) and match outcomes (`result`).

## Data Cleaning Steps

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

## Univariate Analysis

A histogram was created to visualize the distribution of the `golddiffat25` column. The gold difference at 25 minutes shows a roughly normal distribution centered around zero, with most values falling between -5,000 and 5,000. This distribution suggests that most games are balanced around the 25-minute mark, with extreme gold advantages or disadvantages being relatively rare. This finding highlights the competitive nature of League of Legends matches and suggests that small gold advantages can significantly influence outcomes.

![Distribution of Gold Difference](images/distribution_plot.png)

## Bivariate Analysis

The relationship between `golddiffat25` and `result` was analyzed using a scatter plot and a box plot. The scatter plot demonstrated a moderate positive trend, indicating that higher gold differences are generally associated with winning outcomes. The box plot showed a clear distinction in the gold difference distributions for wins and losses. Teams that win matches (`result = 1`) tend to have positive gold differences, while teams that lose (`result = 0`) tend to have negative or neutral gold differences. This analysis confirms that early-game gold leads play a critical role in determining match outcomes.

![Gold Difference vs Match Outcome](images/bivariate_plot.png)

## Aggregate Statistics

The dataset was grouped by match outcome (`result`), and the mean gold difference for each outcome was calculated. Teams that won their matches had an average gold lead of 1,511.96, while losing teams had an average gold deficit of -1,511.96. This significant difference reinforces the idea that gold advantage at 25 minutes is a key indicator of success in League of Legends matches.

### Grouped Table

```plaintext
   result  golddiffat25
0       0  -1511.962828
1       1   1511.962828
```

## Missing Value Handling

Missing values in the `golddiffat25` column were handled by dropping rows where this column was `NaN`. This approach ensured that only complete and accurate data points were used in the analysis. Imputation was not performed because the gold difference is a key feature, and approximating its values could have introduced bias into the results.

## Conclusion

This analysis highlights the importance of early-game gold advantages in League of Legends. The findings suggest that teams with a higher gold difference at 25 minutes are significantly more likely to win. This insight provides valuable strategic implications for teams and fans alike, emphasizing the importance of early-game performance in professional play.


## Step 3: Framing a Prediction Problem
# Prediction Problem

### Problem Statement
The prediction problem involves determining whether a team will win (`result = 1`) or lose (`result = 0`) a League of Legends match based on in-game metrics available at the 25-minute mark. This is a **binary classification problem**, as the response variable (`result`) has two possible outcomes: win or loss. 

### Response Variable
The response variable is `result`, which indicates whether a team won (`1`) or lost (`0`) the match. This variable was chosen because it represents the ultimate outcome of the game, and understanding how early-game metrics impact this outcome can provide valuable insights for strategy development.

### Features
The features used to train the model include:
1. **`golddiffat25`**: Gold difference at the 25-minute mark (quantitative).  
2. **`xpdiffat25`**: Experience difference at the 25-minute mark (quantitative).  
3. **`killsat25`**: Number of kills achieved by the team at the 25-minute mark (quantitative).  
4. **`deathsat25`**: Number of deaths suffered by the team at the 25-minute mark (quantitative).  

These features were chosen because they represent in-game performance metrics available at the time of prediction. They are critical indicators of early-game dominance, which strongly influences match outcomes.

### Metric for Model Evaluation
The primary metric used to evaluate the model is the **F1-score**, which balances precision and recall. This metric was chosen over accuracy because of the potential imbalance in the dataset, where some outcomes (e.g., wins or losses) may occur more frequently than others. The F1-score ensures that the model performs well in predicting both classes without favoring the majority class.

### Justification for Features
All features included in the model are based on information available at the 25-minute mark, ensuring they are valid for prediction. This aligns with the temporal constraints of the problem, as only data up to 25 minutes into the game is used to predict the outcome. Metrics that would only be available after the match ends (e.g., final kills or gold totals) were excluded to avoid data leakage.

### Why This Problem Matters
This prediction problem provides actionable insights into the importance of early-game performance in competitive League of Legends. By predicting match outcomes based on early-game metrics, teams can identify critical areas to improve their strategies and optimize their chances of success.

# Baseline Model Description

### Model Description
The baseline model used is a **Logistic Regression Classifier** implemented within a scikit-learn `Pipeline`. Logistic regression is well-suited for this binary classification task as it predicts the probability of a team winning (`result = 1`) or losing (`result = 0`). The pipeline ensures all preprocessing steps, such as feature scaling, are integrated seamlessly with model training.

### Features in the Model
The model uses the following features:

1. **`golddiffat25`** (Quantitative):  
   This feature represents the gold difference at the 25-minute mark, providing a measure of the team’s economic advantage or disadvantage during the early to mid-game phase.

2. **`xpdiffat25`** (Quantitative):  
   This feature represents the experience difference at the 25-minute mark, giving insight into the team’s relative level advantage or deficit compared to their opponent.

Both features are quantitative and continuous. Since there were no ordinal or nominal features, encoding was not required. The features were scaled using **StandardScaler** within the pipeline to normalize their ranges and ensure improved logistic regression performance.

### Model Performance
The model was evaluated on a test set comprising 30% of the dataset to assess its generalization capability. The following metrics were observed:

- **Accuracy**: 75.56%  
- **Precision**: 76% (Class 0), 75% (Class 1)  
- **Recall**: 75% (Class 0), 76% (Class 1)  
- **F1 Score**: 75% (Class 0), 76% (Class 1)  

These results indicate that the baseline model performs reasonably well for the task, achieving a balanced performance across both classes. The F1-score, in particular, highlights that the model maintains a good balance between precision and recall.

### Is the Model “Good”?
The current baseline model is "good" as an initial attempt because:

1. **Domain Relevance**:  
   The selected features, `golddiffat25` and `xpdiffat25`, are well-aligned with domain knowledge in League of Legends. These metrics are critical indicators of early-game dominance and are readily available at the 25-minute mark.

2. **Balanced Performance**:  
   The model performs consistently across both classes, as seen from the similar F1-scores for wins and losses. The accuracy of 75.56% is also a strong starting point for a baseline model.

3. **Simplicity and Interpretability**:  
   Logistic regression provides clear and interpretable results, making it easier to understand the impact of features on match outcomes.

### Model Limitations
While the baseline model provides a solid starting point, it has the following limitations:
- It does not account for complex interactions or nonlinear relationships between features.
- It uses only two features, which may oversimplify the problem. Adding more features (e.g., kills, deaths, or map objectives) could improve predictive performance.
- Logistic regression may not capture the full complexity of match dynamics, so exploring more advanced models, such as Random Forests or Gradient Boosting, is a logical next step.

### Next Steps
To improve the model, additional features such as kill counts, deaths, and team objectives could be incorporated. Additionally, experimenting with more complex models and feature selection techniques could enhance performance further.

'''
