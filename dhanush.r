# Read student data from a local CSV file
student_features <- read.csv

# Check the first few rows of the data
head(student_features)

# Perform dimensionality reduction with PCA
student_data <- student_features[, c("studied_credits", "num_of_prev_attempts")]

# Scale the data (recommended for PCA)
scaled_data <- scale(student_data)

# Perform PCA and summarize the explained variance
pca <- prcomp(scaled_data, center = TRUE, scale. = TRUE)

# Proportion of variance explained by each principal component
prop_variance <- pca$sdev^2 / sum(pca$sdev^2)

# Scree plot
plot(prop_variance, type = "b", xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     main = "Scree Plot for PCA")

# How much variance is explained by the first two principal components?
first_two_explained <- sum(prop_variance[1:2]) * 100
cat("Proportion of variance explained by the first two components:", first_two_explained, "%\n")

# Cluster the data using KMeans
library(cluster)
kmeans_results <- kmeans(scaled_data, centers = 3, nstart = 20)

# Add cluster labels to the data frame
student_features$kmeans_cluster <- kmeans_results$cluster

# Cluster the data using Hierarchical clustering (Ward's method)
hclust_results <- hclust(dist(scaled_data), method = "ward.D2")

# Visualize clusters using scree plot and dendrogram for hierarchical clustering

# Dendrogram
plot(hclust_results, main = "Hierarchical Clustering Dendrogram",
     xlab = "Students", ylab = "Distance")

# Interpretation
cat("\nProportion of variance explained by the first two components:", first_two_explained, "%\n")

# Explore the distribution of students within clusters for both KMeans and hierarchical clustering. 
# KMeans
cat("\nNumber of students in each KMeans cluster:\n")
table(student_features$kmeans_cluster)

# Hierarchical Clustering
# Cut the dendrogram into clusters
hierarchical_clusters <- cutree(hclust_results, k = 3)
cat("\nNumber of students in each hierarchical cluster:\n")
table(hierarchical_clusters)

# Consider the interpretability of the features associated with each principal component 
# Loadings of the first two principal components
loadings <- pca$rotation[, 1:2]
colnames(loadings) <- c("PC1", "PC2")
rownames(loadings) <- c("studied_credits", "num_of_prev_attempts")
print("\nLoadings of the first two principal components:")
print(loadings)

# Plotting the data with cluster information
library(ggplot2)

# KMeans
ggplot(student_features, aes(x = studied_credits, y = num_of_prev_attempts, color = factor(kmeans_cluster))) +
  geom_point() +
  labs(title = "KMeans Clustering", x = "Studied Credits", y = "Number of Previous Attempts")

# Hierarchical clustering
ggplot(student_features, aes(x = studied_credits, y = num_of_prev_attempts, color = factor(hierarchical_clusters))) +
  geom_point() +
  labs(title = "Hierarchical Clustering", x = "Studied Credits", y = "Number of Previous Attempts")
