# -*- coding: utf-8 -*-
"""Untitled55.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uxtndCTp26a-TpZ5_N9WYEf2H0_OC1YT
"""

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("CustomerChurn").getOrCreate()

data_path = '/content/drive/MyDrive/churn-bigml-80.csv'
df = spark.read.csv(data_path, header=True, inferSchema=True)

df.show(5)
df.printSchema()



df.groupBy('Churn').count().show()

df.describe().show()

from pyspark.sql.functions import col, when, count
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

df.columns

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

df = df.withColumn("Churn", col("Churn").cast("string"))

categorical_cols = ['State', 'International plan', 'Voice mail plan']

indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categorical_cols]

encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_vec") for col in categorical_cols]

numeric_cols = ['Account length', 'Number vmail messages', 'Total day minutes',
                'Total day calls', 'Total day charge', 'Total eve minutes', 'Total eve calls',
                'Total eve charge', 'Total night minutes', 'Total night calls', 'Total night charge',
                'Total intl minutes', 'Total intl calls', 'Total intl charge', 'Customer service calls']

assembler = VectorAssembler(inputCols=[col + "_vec" for col in categorical_cols] + numeric_cols, outputCol="features")

label_indexer = StringIndexer(inputCol="Churn", outputCol="label")

pipeline = Pipeline(stages=indexers + encoders + [assembler, label_indexer])

df_prepared = pipeline.fit(df).transform(df)

train_data, test_data = df_prepared.randomSplit([0.8, 0.2], seed=42)

df_prepared.select('features', 'label').show(5, truncate=False)

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")

dt = DecisionTreeClassifier(featuresCol='features', labelCol='label')

dt_model = dt.fit(train_data)

dt_predictions = dt_model.transform(test_data)

dt_auc = evaluator.evaluate(dt_predictions)
print(f"Decision Tree AUC: {dt_auc}")

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=10)

rf_model = rf.fit(train_data)

rf_predictions = rf_model.transform(test_data)

rf_auc = evaluator.evaluate(rf_predictions)
print(f"Random Forest AUC: {rf_auc}")

from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(featuresCol='features', labelCol='label', maxIter=10)
gbt_model = gbt.fit(train_data)
gbt_predictions = gbt_model.transform(test_data)
gbt_auc = evaluator.evaluate(gbt_predictions)
print(f"Gradient-Boosted Trees AUC: {gbt_auc}")

from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes(featuresCol='features', labelCol='label')
nb_model = nb.fit(train_data)
nb_predictions = nb_model.transform(test_data)
nb_auc = evaluator.evaluate(nb_predictions)
print(f"Naive Bayes AUC: {nb_auc}")

# Convert Spark DataFrame to Pandas for easy plotting
df_pandas = df.select("Churn").toPandas()

# Plot churn distribution
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
df_pandas['Churn'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.xlabel('Churn')
plt.ylabel('Count')
plt.title('Churn Distribution')
plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
plt.show()

from pyspark.ml.feature import VectorAssembler

# Define the features to be used for clustering
features = [
    'Account length', 'Area code', 'Number vmail messages', 'Total day minutes',
    'Total day calls', 'Total day charge', 'Total eve minutes', 'Total eve calls',
    'Total eve charge', 'Total night minutes', 'Total night calls', 'Total night charge',
    'Total intl minutes', 'Total intl calls', 'Total intl charge', 'Customer service calls'
]

# Assemble the features into a single vector
assembler = VectorAssembler(inputCols=features, outputCol='features')
df_features = assembler.transform(df)

from pyspark.ml.clustering import KMeans

# Initialize the K-Means model
kmeans = KMeans(featuresCol='features', predictionCol='cluster', k=3, seed=42)  # Using k=3 clusters

# Fit the K-Means model
kmeans_model = kmeans.fit(df_features)

# Make predictions
kmeans_predictions = kmeans_model.transform(df_features)

# Show cluster assignments for a few records
kmeans_predictions.select('cluster').show(5)

from pyspark.ml.evaluation import ClusteringEvaluator

# Initialize the Clustering Evaluator with the Silhouette metric
evaluator = ClusteringEvaluator(predictionCol='cluster', featuresCol='features', metricName='silhouette')

# Calculate the Silhouette Score
silhouette_score = evaluator.evaluate(kmeans_predictions)
print(f"Silhouette Score for K-Means Clustering: {silhouette_score:.4f}")

import pandas as pd  # Import Pandas
from pyspark.ml.feature import PCA
import matplotlib.pyplot as plt

# Step 1: Apply PCA to reduce the features to 2 dimensions
pca = PCA(k=2, inputCol='features', outputCol='pca_features')
pca_model = pca.fit(kmeans_predictions)
pca_result = pca_model.transform(kmeans_predictions)

# Step 2: Convert the PCA results to a Pandas DataFrame for visualization
pca_data = pca_result.select('pca_features', 'cluster').toPandas()

# Step 3: Extract the PCA components for plotting
pca_data[['PCA1', 'PCA2']] = pd.DataFrame(pca_data['pca_features'].tolist(), index=pca_data.index)

# Step 4: Plot the PCA results
plt.figure(figsize=(10, 6))
plt.scatter(pca_data['PCA1'], pca_data['PCA2'], c=pca_data['cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Visualization of K-Means Clustering')
plt.colorbar(label='Cluster')
plt.show()

from pyspark.ml.evaluation import ClusteringEvaluator

# List to store Silhouette scores for different values of K
silhouette_scores = []

# Try different values of K (e.g., from 2 to 10)
for k in range(2, 11):
    kmeans = KMeans(k=k, seed=1, featuresCol='features', predictionCol='cluster')
    kmeans_model = kmeans.fit(df_features)
    predictions = kmeans_model.transform(df_features)

    # Evaluate the clustering using Silhouette score
    evaluator = ClusteringEvaluator(predictionCol='cluster', featuresCol='features', metricName='silhouette')
    silhouette = evaluator.evaluate(predictions)
    silhouette_scores.append((k, silhouette))
    print(f"Silhouette Score for K={k}: {silhouette:.4f}")

# Find the best K
best_k = max(silhouette_scores, key=lambda x: x[1])[0]
print(f"Best K based on Silhouette score: {best_k}")

import pickle # import the pickle module



with open('model.pkl', 'wb') as f:
    pickle.dump(model, f) # Now pickle.dump should work correctly

!ls

