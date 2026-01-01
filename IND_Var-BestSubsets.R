#Best Subsets Optimization
# https://www.statology.org/regsubsets-in-r/
install.packages("leaps")
install.packages("ISLR")
library(leaps)
library(ISLR)

#Dataset
Hitters

best_sub = regsubsets(Salary ~ .,data = Hitters,nvmax = 19,method = "forward")
summary(best_sub) #  Asteriks show which 1, 2, etc. are best

lrmMod <- lm(Hitters$Salary ~ Hitters$CRBI +
                Hitters$Hits
             +  Hitters$PutOuts
              , data = Hitters)

#Predict Y on Test Dataset
pred_ChDep <- predict(lrmMod, Hitters, type="response") 

#Figure out how to apply this, see how many VARs should be in model
#Calculate evaluation metrics
mse <- mean((test$medv - pred_medv)^2)
mae <- mean(abs(test$medv - pred_medv))
rmse <- sqrt(mse)
r2 <- summary$r.squared

# Print evaluation metrics
cat("MAE:", mae, "\n", "MSE:", mse, "\n", 
    "RMSE:", rmse, "\n", "R-squared:", r2, "\n")

