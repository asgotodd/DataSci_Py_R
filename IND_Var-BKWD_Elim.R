# Ref: https://www.geeksforgeeks.org/machine-learning/ml-multiple-linear-regression-backward-elimination-technique/
#dataset
mtcars


intercept_only <- lm(mpg ~ 1,data = mtcars)
all <- lm(mpg ~ .,data = mtcars)

# In backward optimization, start with the "All" model and remove vars
backward <- step(all,direction = 'backward')
#each step shows 1 var being removed
#Select the version with overall LOWEST AIC (the last one)
# this optimization says keep am, qsec, wt

backward$anova

