
# Personal Project Apache Spark with dataset of Amazon Metadata

This project demonstrates the application of **Apache Spark** for machine learning tasks, using its **MLlib** library for distributed data processing and model training. The project showcases the use of classification algorithms with Spark's DataFrame API and provides a detailed evaluation of the models.

## Technologies and Libraries

- **Apache Spark 3.x**: Used for distributed data processing and machine learning tasks via the MLlib library.
- **PySpark**: Python API for Spark, enabling large-scale parallel data processing.
- **MLlib**: Spark's scalable machine learning library used for the implementation of classification algorithms.
- **Gradient Boosted Trees (GBT)**: A powerful ensemble algorithm that combines multiple weak learners to create a strong model.

## Key Components

### 1. Data Preprocessing
Data is loaded and preprocessed using Spark DataFrames. The features are vectorized, and the labels are encoded to match the format expected by Spark MLlib algorithms. The key steps include:
- **Splitting Data**: The dataset is split into training and test sets using an 80-20 ratio.
- **Vectorization**: Features are transformed into a vector format using `VectorAssembler` to prepare for machine learning model training.

### 2. Machine Learning Models

#### 2.1 Logistic Regression
- The **Logistic Regression** model is applied using `LogisticRegression` from Spark's MLlib.
- The model is trained on the preprocessed dataset.
- **Accuracy** is calculated using `MulticlassClassificationEvaluator` after testing on the test dataset.

#### 2.2 Gradient Boosted Trees (GBT)
- The **GBTClassifier** is used for training an ensemble model based on decision trees.
- GBT combines multiple weak learners to build a robust model.
- The final model is evaluated on the test data to compute its accuracy.

### 3. Model Evaluation
- The performance of each model is evaluated using the **MulticlassClassificationEvaluator** with the metric set to accuracy.
- Both Logistic Regression and Gradient Boosted Trees models are tested, and their accuracy is computed to measure performance.

## How to Run

1. Set up Apache Spark and PySpark environment.
2. Load the notebook and run each cell sequentially.
3. The models will be trained, and the evaluation results will be displayed.