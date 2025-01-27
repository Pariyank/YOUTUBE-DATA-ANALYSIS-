install.packages(c("rpart", "neuralnet", "kernlab", "ggplot2", "rmarkdown","randomForest"))

library(tidyverse)
library(caret)
library(class)
library(e1071)
library(rpart)
library(neuralnet)
library(kernlab)
library(ggplot2)
library(randomForest)

youtube_data <- read.csv("D:\\SEM 5\\INT 234---37\\YOUTUBE DATASET\\youtube_channel_real_performance_analytics.csv")

youtube_data$Video.Publish.Time <- as.Date(youtube_data$Video.Publish.Time)

youtube_data <- youtube_data %>%
  select(Views, Watch.Time..hours., Subscribers, Estimated.Revenue..USD., Impressions) %>%
  na.omit()

normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

youtube_data_norm <- as.data.frame(lapply(youtube_data, normalize))
numerical_data <- youtube_data_norm %>% select(-Subscribers)
youtube_data_norm$Subscribers <- as.factor(ifelse(youtube_data$Subscribers > mean(youtube_data$Subscribers), "High", "Low"))

set.seed(123)
trainIndex <- createDataPartition(youtube_data_norm$Subscribers, p = 0.8, list = FALSE)
train_data <- youtube_data_norm[trainIndex, ]
test_data <- youtube_data_norm[-trainIndex, ]

metrics <- data.frame(Model = character(), Accuracy = numeric(), stringsAsFactors = FALSE)

plot_results <- function(test_actual, test_predicted, model_name) {
  data <- data.frame(Actual = test_actual, Predicted = test_predicted)
  ggplot(data, aes(x = Actual, y = Predicted)) +
    geom_jitter(color = "blue", alpha = 0.6) +
    theme_minimal() +
    labs(
      title = paste("Actual vs Predicted -", model_name),
      x = "Actual Values",
      y = "Predicted Values"
    )
}
# k-Nearest Neighbors (KNN)
knn_model <- knn(
  train = train_data[, -3],
  test = test_data[, -3],
  cl = train_data$Subscribers,
  k = 5
)
knn_accuracy <- mean(knn_model == test_data$Subscribers)
metrics <- rbind(metrics, data.frame(Model = "KNN", Accuracy = knn_accuracy))
print(paste("Accuracy of KNN:", round(knn_accuracy * 100, 2), "%"))
plot_results(test_data$Subscribers, knn_model, "KNN")

# Naive Bayes
naive_bayes_model <- naiveBayes(Subscribers ~ Views + Watch.Time..hours., data = train_data)
naive_bayes_pred <- predict(naive_bayes_model, test_data)
naive_bayes_accuracy <- mean(naive_bayes_pred == test_data$Subscribers)
metrics <- rbind(metrics, data.frame(Model = "Naive Bayes", Accuracy = naive_bayes_accuracy))
print(paste("Accuracy of Naive Bayes:", round(naive_bayes_accuracy * 100, 2), "%"))
plot_results(test_data$Subscribers, naive_bayes_pred, "Naive Bayes")

# Decision Tree
decision_tree <- rpart(
  Subscribers ~ Views + Watch.Time..hours.,
  data = train_data,
  method = "class"
)
decision_tree_pred <- predict(decision_tree, test_data, type = "class")
decision_tree_accuracy <- mean(decision_tree_pred == test_data$Subscribers)
metrics <- rbind(metrics, data.frame(Model = "Decision Tree", Accuracy = decision_tree_accuracy))
print(paste("Accuracy of Decision Tree:", round(decision_tree_accuracy * 100, 2), "%"))
plot_results(test_data$Subscribers, decision_tree_pred, "Decision Tree")

# SVM
svm_model <- ksvm(Subscribers ~ Views + Watch.Time..hours., data = train_data, kernel = "rbfdot")
svm_pred <- predict(svm_model, test_data)
svm_accuracy <- mean(svm_pred == test_data$Subscribers)
metrics <- rbind(metrics, data.frame(Model = "SVM", Accuracy = svm_accuracy))
print(paste("Accuracy of SVM:", round(svm_accuracy * 100, 2), "%"))
plot_results(test_data$Subscribers, svm_pred, "SVM")

# Neural Network
neural_model <- neuralnet(
  as.numeric(Subscribers) ~ Views + Watch.Time..hours.,
  data = train_data,
  hidden = c(5, 3),
  linear.output = FALSE
)
neural_pred <- compute(neural_model, test_data[, -3])$net.result
neural_pred_class <- ifelse(neural_pred > 0.5, "High", "Low")
neural_pred_class <- factor(neural_pred_class, levels = levels(test_data$Subscribers))  # Ensure levels match
neural_accuracy <- mean(neural_pred_class == test_data$Subscribers)
metrics <- rbind(metrics, data.frame(Model = "Neural Network", Accuracy = neural_accuracy))
print(paste("Accuracy of Neural Network:", round(neural_accuracy * 100, 2), "%"))
plot_results(test_data$Subscribers, neural_pred_class, "Neural Network")

#K-means clustering
kmeans_model <- kmeans(numerical_data, centers = 3, nstart = 25)
youtube_data_norm$Cluster <- as.factor(kmeans_model$cluster)
kmeans_model

#Simple Linear Regression
linear_model <- lm(Views ~ Watch.Time..hours., data = train_data)
linear_pred <- predict(linear_model, test_data)
# Calculate R-squared for performance evaluation
r_squared <- summary(linear_model)$r.squared
print(paste("R-squared for Linear Regression:", round(r_squared, 3)))

#Random Forest
random_forest_model <- randomForest(
  Subscribers ~ Views + Watch.Time..hours.,
  data = train_data,
  ntree = 500, # Number of trees in the forest
  mtry = 2,    # Number of variables considered for splitting at each node
  importance = TRUE # To calculate variable importance
)

print(random_forest_model)

rf_pred <- predict(random_forest_model, test_data)
rf_accuracy <- mean(rf_pred == test_data$Subscribers)
metrics <- rbind(metrics, data.frame(Model = "Random Forest", Accuracy = rf_accuracy))
print(paste("Accuracy of Random Forest:", round(rf_accuracy * 100, 2), "%"))

hc_data <- youtube_data_norm %>% select(-Subscribers)

# Compute Distance Matrix
# Using Euclidean distance
distance_matrix <- dist(hc_data, method = "euclidean")

# Hierarchical Clustering
# Using the Ward's method for clustering
hc_model <- hclust(distance_matrix, method = "ward.D2")

# Cut the Tree into Clusters
# Specify the number of clusters
num_clusters <- 3
hc_clusters <- cutree(hc_model, k = num_clusters)

# Add Cluster Assignments to the Dataset
youtube_data_norm$Cluster <- as.factor(hc_clusters)

# Visualize Clusters
library(ggplot2)
ggplot(youtube_data_norm, aes(x = Views, y = Watch.Time..hours., color = Cluster)) +
  geom_point(alpha = 0.6, size = 3) +
  theme_minimal() +
  labs(
    title = "Hierarchical Clustering Results",
    x = "Views",
    y = "Watch Time (hours)",
    color = "Cluster"
  )




# Accuracy Comparison Chart
ggplot(metrics, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Accuracy Comparison of Models", x = "Model", y = "Accuracy")

