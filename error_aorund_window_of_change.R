# Example Data
window_size <- c(2, 3, 4, 11)
errors_at_0 <- c(23, 17, 6, 9)

# Create a data frame
data <- data.frame(window_size, errors_at_0)

# Fit the Poisson regression model
model <- glm(errors_at_0 ~ window_size, family = poisson(), data = data)

# Print the summary of the model
summary(model)

# Predicting and plotting the fitted model
data$predicted <- predict(model, type = "response")

# Plotting
library(ggplot2)
ggplot(data, aes(x = window_size, y = errors_at_0)) +
  geom_point(color = "blue") +
  geom_line(aes(y = predicted), color = "red", linetype = "dashed") +
  labs(title = "Poisson Regression of Errors at 0 vs. Window Size",
       x = "Window Size", y = "Number of Errors at 0") +
  theme_minimal()
