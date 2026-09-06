# Baseline Models Documentation

This documentation provides details regarding the dataset preprocessing, model evaluation, and performance comparison of the baseline models located in this directory.

## Dataset Features and Preprocessing

During the model training phase, all zero-variance rows were removed from the dataset. The following features were identified as constant (having a variance of zero) and subsequently dropped:

| Column | Constant Value |
|--------|---------------:|
| **P102** | 1 |
| **P201** | 1 |
| **P202** | 1 |
| **P204** | 1 |
| **P206** | 1 |
| **P401** | 1 |
| **P403** | 1 |
| **P404** | 1 |
| **P502** | 1 |
| **P601** | 1 |
| **P603** | 1 |

Additionally, a static `if` statement was applied to the data before passing it to the model for training and inference.

## Model Evaluation

The baseline models were evaluated on the test dataset. Below are the Confusion Matrices and Classification Reports for each model.

### 1. Isolation Forest
**Confusion Matrix:**
```
[[388764   6534]
 [ 19477  35144]]
```
**Classification Report:**
```
              precision    recall  f1-score   support

         0.0     0.9523    0.9835    0.9676    395298
         1.0     0.8432    0.6434    0.7299     54621

    accuracy                         0.9422    449919
   macro avg     0.8978    0.8134    0.8488    449919
weighted avg     0.9390    0.9422    0.9388    449919
```

### 2. USAD
**Confusion Matrix:**
```
[[395094    204]
 [ 12260  42361]]
```
**Classification Report:**
```
              precision    recall  f1-score   support

         0.0     0.9699    0.9995    0.9845    395298
         1.0     0.9952    0.7755    0.8718     54621

    accuracy                         0.9723    449919
   macro avg     0.9826    0.8875    0.9281    449919
weighted avg     0.9730    0.9723    0.9708    449919
```

### 3. VAE (Variational Autoencoder)
**Confusion Matrix:**
```
[[395132    166]
 [ 21953  32668]]
```
**Classification Report:**
```
              precision    recall  f1-score   support

         0.0     0.9474    0.9996    0.9728    395298
         1.0     0.9949    0.5981    0.7471     54621

    accuracy                         0.9508    449919
   macro avg     0.9712    0.7988    0.8599    449919
weighted avg     0.9531    0.9508    0.9454    449919
```

### 4. LSTM-Autoencoder
**Confusion Matrix:**
```
[[388764   6534]
 [ 19477  35144]]
```
**Classification Report:**
```
              precision    recall  f1-score   support

         0.0     0.9523    0.9835    0.9676    395298
         1.0     0.8432    0.6434    0.7299     54621

    accuracy                         0.9422    449919
   macro avg     0.8978    0.8134    0.8488    449919
weighted avg     0.9390    0.9422    0.9388    449919
```

### 5. Autoencoder
**Confusion Matrix:**
```
[[371275  24023]
 [  7649  46972]]
```
**Classification Report:**
```
              precision    recall  f1-score   support

         0.0     0.9798    0.9392    0.9591    395298
         1.0     0.6616    0.8600    0.7479     54621

    accuracy                         0.9296    449919
   macro avg     0.8207    0.8996    0.8535    449919
weighted avg     0.9412    0.9296    0.9334    449919
```

## Performance Analysis

Upon analyzing the evaluation metrics across all baseline models, **USAD** outperforms the rest significantly.

### Why USAD Performed Well
- **Highest Accuracy & F1-Score**: USAD achieves the highest overall accuracy (97.23%) and the best F1-score for the anomaly class (87.18%). 
- **Robust Precision-Recall Balance**: In highly imbalanced datasets like this one (where normal data heavily outweighs anomalies), false positives can be problematic. USAD maintains an exceptional precision for anomalies (99.52%) with very few false alarms (only 204 false positives out of 395,298 normal samples) while still keeping a relatively strong recall (77.55%).
- **Adversarial Training**: The architecture of USAD relies on adversarial training which allows it to isolate normal data representations more effectively, capturing the underlying data distribution comprehensively compared to standard autoencoders.

### Why Other Models Failed to Match USAD
- **Autoencoder**: While achieving the highest recall for anomalies (86.00%), it struggled immensely with precision (66.16%). It generated a massive number of false positives (24,023), proving it is overly sensitive and not viable for robust deployment where false alarms are costly.
- **VAE**: The Variational Autoencoder had almost perfect precision (99.49%) but poor recall (59.81%). It missed a large number of actual anomalies (21,953 false negatives).
- **Isolation Forest and LSTM-Autoencoder**: Both models yielded identical and mediocre performance metrics (72.99% F1-score for anomalies). They failed to balance precision and recall effectively, missing many anomalies while also producing a noticeable amount of false positives.

The model selected for deployment: USAD
