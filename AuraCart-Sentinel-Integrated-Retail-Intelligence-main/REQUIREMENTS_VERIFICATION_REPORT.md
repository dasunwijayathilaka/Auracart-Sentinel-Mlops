# AuraCart Sentinel ML Project — Requirements Verification Report
**Date:** April 4, 2026  
**Project:** ITS 2140 Machine Learning Group Project  
**Status:** ✅ LARGELY COMPLETE WITH SOME GAPS

---

## Executive Summary
Your implementation has successfully **fulfilled most core requirements** of the ITS 2140 capstone project. The system achieves **excellent performance on customer segment classification (100% accuracy)** and includes comprehensive MLflow tracking and cloud deployment infrastructure. However, there are notable gaps in:
1. **Delivery Status Classification** — only 19.2% accuracy (needs urgently addressed)
2. **Unsupervised Clustering (Task 3.5)** — not implemented
3. **Vertex AI deployment** — infrastructure is set up, but endpoint testing incomplete

---

## PART 1: PROJECT OVERVIEW — ✅ COMPLETED

| Aspect | Status | Notes |
|--------|--------|-------|
| Business Context Understanding | ✅ | Project correctly framed as AuraCart analytics engine |
| Three simultaneous tasks identified | ✅ | Regression (price), Classification (segment+status), Clustering |
| MLflow integration mandate | ✅ | Fully implemented with experiment tracking |
| Cloud deployment on Vertex AI | ⚠️ | Setup complete, but limited endpoint testing documented |

---

## PART 2: DATA ASSET — ✅ COMPLETED

### 2.1 Dataset Overview
| Item | Status | Your Result |
|------|--------|-------------|
| Load from Hugging Face | ✅ | Dataset loaded correctly from `millat/e-commerce-orders` |
| Shape and records | ✅ | 10,000 records loaded; 23 features after preprocessing |
| Key challenges identified | ✅ | Class imbalance documented and addressed |

### 2.2 Class Imbalance & Characteristics
| Feature | Specification | Your Dataset |
|---------|--------------|-------------|
| **Delivery Status** | Delivered 70%, Shipped 20%, Pending 5%, Returned 5% | ✅ Confirmed in confusion matrix analysis |
| **Customer Segment** | New 50%, Returning 35%, VIP 15% | ✅ Confirmed distribution shown |
| **Price Range** | $5–$500 | ✅ Correctly scaled |
| **Continuous Features** | Need scaling | ✅ StandardScaler applied |
| **Categorical Features** | Need One-Hot Encoding | ✅ OneHotEncoder applied |

---

## PART 3: PHASE 1 — FOUNDATIONAL DEVELOPMENT

### Task 3.1: Exploratory Data Analysis (EDA) & Preprocessing Pipeline

#### Requirements Checklist:
| Requirement | Status | Implementation |
|------------|--------|-----------------|
| **1. Continuous Variables Analysis** | ✅ | Histograms and distributions analyzed in Notebook 1 |
| **2. Feature Correlation Analysis** | ✅ | Correlation matrix generated; multicollinearity assessed |
| **3. Categorical Variable Analysis** | ✅ | Distribution confirmed for customer_segment and delivery_status |
| **4. Key Findings Summary** | ✅ | EDA section documents all major insights |
| **Preprocessing Pipeline (Scikit-learn)** | ✅ | |
| — Categorical Encoding | ✅ | One-Hot Encoding implemented for 5 categorical features |
| — Feature Scaling | ✅ | StandardScaler applied to numerical features |
| — Reproducibility | ✅ | ColumnTransformer + Pipeline ensures consistent preprocessing |

**VERDICT:** ✅ **FULLY COMPLETED** — EDA is thorough, preprocessing pipeline is production-ready.

---

### Task 3.2: Continuous Price Prediction Modeling

#### Requirements Checklist:

**1. Multiple Linear Regression Model**
| Item | Requirement | Your Implementation | Result |
|------|-------------|-------------------|--------|
| Model Type | OLS Linear Regression | Pipeline with LinearRegression | ✅ Correct |
| Gradient Descent Experimentation | Vary learning rate, batch size, epochs | SGDRegressor tested with 5 configs | ✅ Implemented |
| Learning Rates Tested | "Important parameters" | 0.001, 0.01, 0.1 | ✅ Done |
| Epochs Tested | "number of times" | 100, 500, 1000 | ✅ Done |
| Batch Size Discussion | "determine how many samples" | Mentioned (SGD uses batch_size=1) | ⚠️ Limited discussion |

**2. Regression Evaluation Metrics**
| Metric | Status | Your Value | Interpretation |
|--------|--------|-----------|-----------------|
| MSE (Mean Squared Error) | ✅ | 14,940.37 | Moderate prediction error squared |
| MAE (Mean Absolute Error) | ✅ | $98.94 | Average error ~$99 per prediction |
| RMSE (Root MSE) | ✅ | $122.23 | More interpretable than MSE |
| Explanation of MSE vs MAE | ✅ | Clearly explained in notebook markdown | ✅ Done |
| Financial Impact Discussion | ✅ | Explains risk of large errors to AuraCart pricing | ✅ Done |

**3. Cross-Validation**
| Item | Requirement | Your Implementation | Result |
|------|-------------|-------------------|--------|
| K-Fold CV | k=5 folds | 5-Fold Cross-Validation applied | ✅ Correct |
| CV MSE Mean | Report average | 14,487.70 (±275.57 std) | ✅ Done |
| CV MAE Mean | Report average | 97.52 (±0.89 std) | ✅ Done |
| Per-fold results | Show variance | All 5 fold MSEs listed | ✅ Done |
| Underfitting/Overfitting Analysis | Discuss bias-variance | **Train MSE ≈ CV MSE → UNDERFITTING** | ⚠️ Partially explained |

**Analysis:** The similar performance on training and CV folds (CV MSE ~14,488 vs test MSE ~14,940) indicates **high bias / underfitting**. This means:
- The linear model is too simple for the data
- Features don't have strong linear relationship with price
- This is realistic for synthetic e-commerce data (pricing is often arbitrary)
- ✅ **This aligns with the synthetic nature of the dataset**

**VERDICT:** ✅ **SUBSTANTIALLY COMPLETED** — All metrics calculated and logged. Minor gap in deeper discussion of batch size impact.

---

### Task 3.3: Multi-Class Classification Modeling

#### Requirements Checklist:

**1. Softmax Regression Model**
| Item | Requirement | Your Implementation | Status |
|------|-------------|-------------------|--------|
| Algorithm | Multinomial Logistic Regression | LogisticRegression(multi_class='multinomial') | ✅ |
| Explanation of why not Linear Reg | "Explain why linear regression..." | Provided in markdown section 4. | ✅ |
| Softmax Function Description | "Convert outputs to probabilities" | Mathematical formula included | ✅ |
| Log-Loss Metric | "categorical cross-entropy loss" | Calculated for all models | ✅ |

**2. Decision Threshold Adjustment**
| Item | Requirement | Your Implementation | Status |
|------|-------------|-------------------|--------|
| Extract Probabilities | "predicted probabilities" | yc_proba = predict_proba() | ✅ |
| Experiment with Thresholds | "different decision thresholds" | Sweep from 0.10 to 0.50 | ✅ |
| Rare Class Focus | "rare Returned delivery status" | Threshold analysis for Returned class | ✅ |
| Precision/Recall Sensitivity | "change sensitivity" | Showed Precision, Recall, F1 vs threshold | ✅ |

**3. Classification Evaluation**
| Item | Requirement | Your Implementation | Status |
|------|-------------|-------------------|--------|
| Confusion Matrix | "Generate and visualize" | Confusion matrix generated + saved as .png | ✅ |
| Analyze Misclassifications | "identify patterns" | Discussed New vs Returning vs VIP confusion | ✅ |
| Business Context | "explain in context" | Discussed VIP detection vs false alarms | ✅ |

**VERDICT:** ✅ **FULLY COMPLETED** — All classification tasks thoroughly implemented.

---

### Task 3.4: Classification Performance Analysis & Risk Evaluation  

#### Requirements Checklist:

**1. Confusion Matrix Analysis**
| Item | Requirement | Your Implementation | Status |
|------|-------------|-------------------|--------|
| Generate Matrix | "visualize confusion matrix" | Generated for both customer_segment and delivery_status | ✅ |
| Examine Distribution | "distributed across classes" | Detailed analysis provided | ✅ |
| Identify Patterns | "misclassification patterns" | Discussed VIP vs Returning confusion | ✅ |

**2. Class-wise Metrics**
| Item | Requirement | Your Implementation | Status |
|------|-------------|-------------------|--------|
| Precision per class | "Precision for every class" | classification_report output included | ✅ |
| Recall per class | "Recall for every class" | classification_report output included | ✅ |
| F1-score per class | "F1-score for every class" | Weighted F1 calculated | ✅ |
| Explanation | "what each metric means" | Clear explanations in markdown sections | ✅ |

**3. Precision-Recall Trade-offs**
| Item | Requirement | Your Implementation | Status |
|------|-------------|-------------------|--------|
| Explain trade-off | "trade-off between precision and recall" | Threshold calibration section shows this | ✅ |
| Business Examples | "examples from dataset" | Returned delivery status discussed | ✅ |
| Business Consequences | "business consequences" | Cost analysis provided | ✅ |

**4. Asymmetric Risk Analysis**
| Item | Requirement | Your Implementation | Status |
|------|-------------|-------------------|--------|
| False Positive vs FN | "Compare FP and FN impact" | Detailed in markdown section 4.5 | ✅ |
| Example Scenarios | "e.g., predicting Returned" | VIP customer misclassification discussed | ✅ |
| Risk Justification | "Justify final model choice" | Cost-sensitive weighting justification provided | ✅ |

**VERDICT:** ✅ **FULLY COMPLETED** — Comprehensive evaluation of model risks and business implications.

---

### Task 3.5: Unsupervised Learning — Customer Behavior Clustering  

#### Requirements Checklist:

| Requirement | Status | Notes |
|------------|--------|-------|
| **K-Means Clustering** | ❌ **NOT IMPLEMENTED** | — |
| **Elbow Method / WCSS** | ❌ **NOT IMPLEMENTED** | — |
| **Silhouette Score** | ❌ **NOT IMPLEMENTED** | — |
| **Cluster Interpretation** | ❌ **NOT IMPLEMENTED** | — |
| **Business Insights from Clusters** | ❌ **NOT IMPLEMENTED** | — |

**🔴 CRITICAL GAP:** This is an entire task worth significant points. K-Means clustering should be performed in [Notebook 3](notebooks/3_unsupervised_clustering.ipynb), but implementation completeness is unclear.

**VERDICT:** ❌ **NOT VERIFIED** — Clustering implementation needs immediate verification/completion.

---

## PART 4: PHASE 2 — PRODUCTION DEPLOYMENT

### Task 4.1: Experiment Tracking with MLflow

#### Requirements Checklist:

| Requirement | Status | Your Implementation |
|------------|--------|-------------------|
| **MLflow Installation** | ✅ | mlflow library installed and configured |
| **Logging Integration** | ✅ | mlflow.start_run() and logging calls throughout |
| **Parameters Tracked** | ✅ | Model type, C, max_iter, learning_rate, epochs logged |
| **Metrics Tracked** | ✅ | Accuracy, F1, Log-loss, MSE, MAE logged |
| **Artifacts Saved** | ✅ | Models logged via mlflow.sklearn.log_model() |
| **Environment Info** | ✅ | Python packages listed in requirements.txt |
| **Screenshots in Report** | ⚠️ | Mention MLflow runs but unclear if UI screenshots included |

**Actual Experiment Runs Tracked:**

| Experiment | Runs | Best Model | Metrics |
|------------|------|-----------|---------|
| **Regression (OLS + SGD)** | 6 total | OLS Base | MSE=14,940, MAE=98.94 |
| **Classification (Segment)** | 20+ total | Cost-Sensitive + Ensemble | Acc=100%, F1=1.0, LL=0.0087 |
| **Classification (Delivery)** | Multiple | Baseline Softmax | Acc=19.2%, F1=0.2455, LL=1.3906 |

**VERDICT:** ✅ **SUBSTANTIALLY COMPLETED** — MLflow integration is comprehensive. Minor issue: report should include explicit MLflow UI screenshots.

---

### Task 4.2: Model Packaging & Artifact Management

#### Requirements Checklist:

| Requirement | Status | Your Implementation |
|------------|--------|-------------------|
| **Select Final Model** | ✅ | Cost-Sensitive RF selected (100% accuracy) |
| **Create Unified Pipeline** | ✅ | ColumnTransformer + Classifier combined |
| **Save as model.joblib** | ✅ | Saved to artifacts/ directory |
| **Capture Dependencies** | ✅ | requirements.txt created |
| **Upload to Google Cloud Storage** | ⚠️ | Script provided; actual upload status unclear |

**Artifact Status:**
```
✅ artifacts/model.joblib              — Final customer_segment model
✅ artifacts/preprocessing_pipeline.joblib  — Preprocessing pipeline
✅ artifacts/requirements.txt          — Dependencies listed
✅ artifacts/conda.yaml                — Environment file
```

**VERDICT:** ✅ **SUBSTANTIALLY COMPLETED** — Artifacts properly serialized. Need confirmation of GCS upload.

---

### Task 4.3: Vertex AI Deployment

#### Requirements Checklist:

| Requirement | Status | Notes |
|------------|--------|-------|
| **Import Model to Vertex AI** | ⚠️ | Script exists; execution status unclear |
| **Select Pre-built Container** | ⚠️ | Scikit-learn container referenced |
| **Deploy to Endpoint** | ⚠️ | Endpoint creation commands documented |
| **Test with JSON Payload** | ⚠️ | Test cases written but deployment verification limited |
| **Screenshot of Endpoint** | ⚠️ | Endpoint status shown but may need refresh |
| **Endpoint Functionality Verified** | ⚠️ | Limited live testing documentation |

**GCloud Commands Attempted:**
```bash
gcloud ai models upload --region=us-central1 ...
gcloud ai endpoints create auracart-deployment ...
gcloud ai endpoints deploy-model 7797297209292095488 ...  # Exit code: 1 (FAILED)
```

**⚠️ ISSUE DETECTED:** Last deployment command failed (exit code 1). This needs investigation:
- Check gcloud authentication
- Verify service account permissions
- Review error logs from failed deployment

**VERDICT:** ⚠️ **INCOMPLETE** — Deployment infrastructure is set up, but endpoint deployment needs debugging and live testing.

---

## FINAL ACCURACY RESULTS SUMMARY

### ✅ Regression Task (Price Prediction)
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Test MAE | $98.94 | < $100 | ✅ Met |
| Test RMSE | $122.23 | — | ✅ Reasonable |
| CV Fold Consistency | ±0.89 MAE | Low variance | ✅ Stable |

**Verdict: ACCEPTABLE** — Given synthetic data without strong pricing signals, this performance is realistic.

---

### ✅✅ Classification Task 1: Customer Segment Prediction (🎯 PRIMARY DEPLOYMENT TARGET)
| Metric | Value | Interpretation | Status |
|--------|-------|-----------------|--------|
| **Test Accuracy** | **100%** | Perfect classification | ✅✅ Excellent |
| **Weighted F1** | **1.0000** | Balanced across all classes | ✅✅ Excellent |
| **Log-Loss** | **0.0087** | Excellent probability calibration | ✅✅ Excellent |
| **Per-Class Recall** | 100% all classes | All segments detected | ✅✅ Excellent |

**Model Selected:** Cost-Sensitive RandomForest with VIP weight = 5x  
**Champion Strategy:** Ensemble voting → Cost-sensitive tuning → Advanced feature engineering  

**Business Impact:**
- ✅ Perfect VIP detection (0 VIPs missed)
- ✅ 100% New customer identification  
- ✅ 100% Returning customer identification  
- ✅ Ready for production deployment

**Verdict: 🏆 EXCEEDS REQUIREMENTS**

---

### ⚠️ Classification Task 2: Delivery Status Prediction
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Test Accuracy** | **19.2%** | ≥ 50% desired | ❌ Far Below Target |
| **Weighted F1** | **0.2455** | ≥ 0.60 desired | ❌ Far Below Target |
| **Log-Loss** | **1.3906** | < 0.5 desired | ❌ Far Below Target |
| **Delivered Class Recall** | ~62% | 70% expected | ⚠️ Below target |
| **Returned Class Recall** | ~14% | 50%+ desired | ❌ Critical gap |

**Root Causes:**
1. **Severe Class Imbalance:** 70% Delivered class dominates
2. **Insufficient Cost-Sensitive Tuning:** Not applied to this task
3. **No SMOTE/Resampling:** Could improve minority class performance
4. **Weak Feature Signals:** Limited features correlated with delivery status

**What Should Be Done:**
- Apply cost-weighted penalties: weight Returned/Pending 5-10x higher
- Implement SMOTE for synthetic minority oversampling
- Use stratified cross-validation throughout
- Consider threshold adjustment (lower threshold for rare classes)
- Try ensemble methods or XGBoost

**Verdict: 🔴 REQUIRES URGENT IMPROVEMENT** — Current 19.2% accuracy is unacceptable for production. This system would perform worse than random guessing (25% on 4 balanced classes).

---

## IMPLEMENTATION CHECKLIST

### Core Requirements Status

| Task | Required | Implemented | Status |
|------|----------|-------------|--------|
| **3.1 EDA & Preprocessing** | ✅ | ✅ Full | ✅ Complete |
| **3.2 Regression Modeling** | ✅ | ✅ Full | ✅ Complete |
| **3.2-1 SGD Experimentation** | ✅ | ✅ 5 configs | ✅ Complete |
| **3.2-2 Cross-Validation** | ✅ | ✅ 5-Fold | ✅ Complete |
| **3.2-3 Evaluation Metrics** | ✅ | ✅ MSE, MAE, RMSE | ✅ Complete |
| **3.3 Classification (Softmax)** | ✅ | ✅ Full | ✅ Complete |
| **3.3-2 Threshold Calibration** | ✅ | ✅ 6 thresholds tested | ✅ Complete |
| **3.4 Performance Analysis** | ✅ | ✅ Full | ✅ Complete |
| **3.4-1 Confusion Matrix** | ✅ | ✅ Generated + visualized | ✅ Complete |
| **3.4-2 Class-wise Metrics** | ✅ | ✅ Precision, Recall, F1 | ✅ Complete |
| **3.4-3 Precision-Recall Trade-offs** | ✅ | ✅ Discussed | ✅ Complete |
| **3.4-4 Asymmetric Risk Analysis** | ✅ | ✅ FP vs FN discussed | ✅ Complete |
| **3.5 K-Means Clustering** | ✅ | ❓ **UNCLEAR** | ⚠️ Verify |
| **3.5-1 Elbow Method** | ✅ | ❓ **UNCLEAR** | ⚠️ Verify |
| **3.5-2 Cluster Interpretation** | ✅ | ❓ **UNCLEAR** | ⚠️ Verify |
| **3.5-3 Business Insights** | ✅ | ❓ **UNCLEAR** | ⚠️ Verify |
| **4.1 MLflow Tracking** | ✅ | ✅ Full | ✅ Complete |
| **4.2 Model Packaging** | ✅ | ✅ Full | ✅ Complete |
| **4.3 Vertex AI Deployment** | ✅ | ⚠️ Partial | ⚠️ Needs debugging |

---

## CRITICAL ISSUES & ACTION ITEMS

### 🔴 CRITICAL
1. **Delivery Status Accuracy (19.2%)** — FAR BELOW acceptable. Must implement cost-sensitive learning immediately.
2. **Clustering Task (3.5) Status Unknown** — Verify implementation in Notebook 3
3. **Vertex AI Deployment Failed** — Last gcloud command failed; endpoint not verified live

### ⚠️ HIGH PRIORITY
4. **Deliv Status → Cost-Sensitive Learning** — Apply weighted penalties
5. **Delivery Status → Class Resampling** — Try SMOTE or undersampling
6. **Endpoint Testing** — Send actual JSON prediction requests to verify functionality

### 📝 MEDIUM PRIORITY
7. **MLflow Screenshots** — Include UI screenshots in final report
8. **Batch Size Discussion** — Expand explanation in SGD section
9. **GCS Upload Verification** — Confirm artifacts uploaded to Cloud Storage

---

## STRENGTHS OF YOUR IMPLEMENTATION

✅ **Exceptional work on customer segment classification** (100% accuracy)  
✅ **Comprehensive evaluation metrics** (MSE, MAE, F1, Log-loss, confusion matrices)  
✅ **Rigorous cross-validation methodology** (5-fold, stratified splits)  
✅ **MLflow integration** (20+ experiments tracked with full reproducibility)  
✅ **Production-ready artifact management** (joblib serialization, requirements.txt)  
✅ **Thorough business context analysis** (asymmetric risk, precision-recall trade-offs)  
✅ **Clear documentation** (markdown explanations throughout notebooks)

---

## SUMMARY & RECOMMENDATIONS

### Current Status
Your project has achieved **strong results on 60-70% of requirements**, with exceptional performance on customer segment classification (the primary deployment target). However, there are critical gaps:

1. **Delivery Status Classification:** 19.2% accuracy is unacceptable
2. **Clustering Task:** Implementation status unclear
3. **Endpoint Deployment:** Needs debugging and live testing

### Recommended Actions (Priority Order)
1. **URGENT:** Fix delivery status model using cost-sensitive learning (target: 50%+ accuracy)
2. **HIGH:** Verify/complete clustering task in Notebook 3
3. **HIGH:** Debug and verify Vertex AI endpoint deployment
4. **MEDIUM:** Add MLflow UI screenshots to final report
5. **MEDIUM:** Expand SGD batch size analysis

### Expected Grade Impact
- **If all issues fixed:** **A / A-** (90-95%) — Excellent capstone project
- **Current incomplete state:** **B+ / B** (80-85%) — Good fundamentals, gaps on delivery + clustering
- **If delivery status left unfixed:** **B- / C+** (70-77%) — Major deduction for unacceptable accuracy

---

**Report Generated:** April 4, 2026  
**Recommendation:** Address critical issues immediately. Your foundation is strong; fixes are achievable.
