
# Personal Project Apache Spark with dataset of Amazon Metadata

This project demonstrates the application of **Apache Spark** for machine learning tasks, using its **MLlib** library for distributed data processing and model training. The project showcases the use of classification algorithms with Spark's DataFrame API and provides a detailed evaluation of the models.

## DataSet
[DataSet Directory Sample MetaData download at here](https://nijianmo.github.io/amazon/index.html)


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

#### 2.1 Decision Tree
- The **Decision Tree Classifier** is applied using `DecisionTreeClassifier` from Spark's MLlib.
- The model is trained on the preprocessed dataset.
- **Accuracy** is calculated using `MulticlassClassificationEvaluator` after testing on the test dataset.

#### 2.2 Random Forest
- The **Random Forest Classifier** is applied using `RandomForestClassifier` from Spark's MLlib.
- Random Forest builds multiple decision trees and merges them to get a more accurate and stable prediction.
- The final accuracy is computed after testing on the test data.

#### 2.3 Logistic Regression
- The **Logistic Regression** model is applied using `LogisticRegression` from Spark's MLlib.
- The model is trained on the preprocessed dataset.
- **Accuracy** is calculated using `MulticlassClassificationEvaluator` after testing on the test dataset.

#### 2.4 Gradient Boosted Trees (GBT)
- The **GBTClassifier** is used for training an ensemble model based on decision trees.
- GBT combines multiple weak learners to build a robust model.
- The final model is evaluated on the test data to compute its accuracy.

### 3. Model Evaluation
- The performance of each model is evaluated using the **MulticlassClassificationEvaluator** with the metric set to accuracy.
- The following models are tested, and their accuracy is computed to measure performance:
  - Decision Tree
  - Random Forest
  - Logistic Regression
  - Gradient Boosted Trees

### 4. Virtual Machine
- VirtualBox Manager
- On Ubuntu/Linux Environment
  
### 5. Installation instructions
  1. Download and install the latest version of Ubuntu.
  2. Setup Apache Spark and Python3 on Ubuntu
    - 2.1 Install Java 
      + Apache Spark requires Java to run, so install OpenJDK:
        `sudo apt update`
        `sudo apt install openjdk-11-jdk`
    Verify the installation: `java -version` or `java -V`
  3. Install Python3 and pip
    If Python3 is not already installed, you can install it using:  `sudo apt update` or `sudo apt install python3 python3-pip`
    Verify the installation: `python3 --version`
  4. Download and Install Apache Spark
    Download the latest version of Spark (adjust the version as necessary, you can find the latest version on the [Apache Spark download at here](https://dlcdn.apache.org/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz)
    Use `wget https://dlcdn.apache.org/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz`
    Extract the downloaded Spark archive: `tar -xvf spark-3.5.3-bin-hadoop3.tgz`
    Move the extracted Spark folder to **phuc03/spark**, before move I also created folder **"Spark"** in **/home/phuc03** : `sudo mv spark-3.5.3-bin-hadoop3 /home/phuc03/spark`
    Set Environment Variables in file .Bashrc => `/home/phuc03` if you don't see, please press **Ctrl H** to display at their. Or use terminal `sudo nano ~/.bashrc`
    Add:
        `export SPARK_HOME=/opt/spark
        export PATH=$SPARK_HOME/bin:$PATH`
    Then, Ctrl X -> Enter -> Enter, To save file .bashrc
    At terminal, we must need `source ~/.bashrc `
  5. Install PySpark and Jupyter NoteBook
    `pip3 install pyspark`
    `sudo pip3 install jupyter`
  6. Start HDFS with Hadoop Master and Hadoop Salve
    **You must make sure to always ensure ssh connection between machines and proper netplan configuration**
    Access the "hadoop" folder in the "bin" file or use terminal `start-dfs.sh` `start-yarn.sh` `start-all.sh`
    Upload DataSet in Browser Directory in **hostname:9870/explorer.html#/**
    Then, Open new Terminal in folder **"/home/phuc03/spark"** : `start-master.sh` `start-salve.sh`
    Turn on Worker with Master and Salve: `/home/phuc03/spark/bin/spark-class org.apache.spark.deploy.worker.Worker spark:master:7077` master is hostname of master machine.
  7. Start Jupyter NoteBook
    `jupyter notebook` => **localhost:8888**
## How to Run
1. Set up Apache Spark and PySpark environment.
2. Load the notebook and run each cell sequentially.
3. The models will be trained, and the evaluation results will be displayed.
4. If you prefer, you can run Spark in local mode without setting up a full cluster. Spark is capable of running on a single machine using the local master URL. This is ideal for development and testing purposes.
   ` spark-submit --master local[*] your_script.py`
# Conclusion
In this project, we applied multiple machine learning algorithms such as Decision Tree, Random Forest, Logistic Regression, and Gradient Boosted Trees using Apache Spark’s MLlib library. This demonstrated the power of Spark for distributed data processing and parallel model training, making it a suitable solution for big data and large-scale machine learning tasks.

By running this notebook, you learned how to preprocess data, train different models, evaluate their accuracy, and use Spark’s distributed computing capabilities. This setup can be extended to more advanced tasks by adjusting parameters, adding more data, or experimenting with different models.

Thank you!
We hope this guide and project helped you better understand Apache Spark's potential for machine learning and distributed computing. Thank you for using this project, and we wish you success with your own data projects!
