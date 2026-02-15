# Machine Learning Model Comparison

## a. Problem Statement
Test

## b. Dataset Description
Test

## c. Models Used and Evaluation Metrics

### Model Performance Comparison


<table border="1" cellspacing="0" cellpadding="6">
    <thead>
        <tr>
            <th>ML Model Name</th>
            <th>Accuracy</th>
            <th>AUC</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>MCC</th>
        </tr>
    </thead>
    <tbody>

        <tr>
            <td>Decision Tree</td>
            <td>0.6476</td>
            <td>0.8791</td>
            <td>0.6482</td>
            <td>0.6475</td>
            <td>0.6474</td>
            <td>0.6241</td>
        </tr>
    
        <tr>
            <td>Naive Bayes</td>
            <td>0.9114</td>
            <td>0.9925</td>
            <td>0.9119</td>
            <td>0.9114</td>
            <td>0.9113</td>
            <td>0.9056</td>
        </tr>
    
        <tr>
            <td>kNN</td>
            <td>0.9869</td>
            <td>0.9948</td>
            <td>0.9869</td>
            <td>0.9869</td>
            <td>0.9869</td>
            <td>0.9860</td>
        </tr>
    
        <tr>
            <td>Logistic Regression</td>
            <td>0.9191</td>
            <td>0.9932</td>
            <td>0.9194</td>
            <td>0.9191</td>
            <td>0.9192</td>
            <td>0.9137</td>
        </tr>
    
        <tr>
            <td>Random Forest (Ensemble)</td>
            <td>0.9773</td>
            <td>0.9941</td>
            <td>0.9773</td>
            <td>0.9773</td>
            <td>0.9773</td>
            <td>0.9758</td>
        </tr>
    
        <tr>
            <td>XGBoost (Ensemble)</td>
            <td>0.9824</td>
            <td>0.9950</td>
            <td>0.9825</td>
            <td>0.9824</td>
            <td>0.9824</td>
            <td>0.9813</td>
        </tr>
    
    </tbody>
</table>


### Model Performance Observations


| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Logistic Regression provides a strong baseline model with stable performance across all metrics. Its high AUC and balanced Precision-Recall indicate reliable predictive performance. However, it is outperformed by non-linear and ensemble models, indicating that the dataset likely contains complex feature interactions. |
| Decision Tree | The Decision Tree model shows the weakest performance among all models, with significantly lower Accuracy, F1, and MCC scores. While the AUC is reasonably high (0.8791), the drop in overall classification metrics suggests overfitting and poor generalization compared to ensemble methods. |
| kNN | kNN achieves the highest overall Accuracy and MCC among all models, indicating superior predictive performance and strong class separation. The consistently high metrics suggest that the dataset benefits from local neighborhood-based decision boundaries. |
| Naive Bayes | Naive Bayes performs strongly with excellent AUC and balanced Precision-Recall values. The high AUC (0.9925) indicates strong class separability, but its overall Accuracy and MCC are lower than kNN and ensemble models, suggesting limitations due to its independence assumption. |
| Random Forest (Ensemble) | Random Forest delivers excellent performance with strong generalization ability. The high Accuracy and MCC demonstrate robustness and reduced overfitting compared to a single Decision Tree. Ensemble learning significantly improves predictive stability. |
| XGBoost (Ensemble) | XGBoost performs extremely well across all metrics, slightly below kNN in Accuracy but showing the highest AUC. Its strong MCC indicates reliable performance even in complex decision boundaries. It demonstrates powerful learning capability through boosting and feature interaction modeling. |