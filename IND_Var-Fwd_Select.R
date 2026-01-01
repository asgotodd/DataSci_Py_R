# Forward Var Select
# Process:  https://www.statology.org/forward-selection/
#  Step 1: Fit an intercept-only regression model with no predictor variables. Calculate the AIC* value for the model.
#  Step 2: Fit every possible one-predictor regression model. Identify the model that produced the lowest AIC and also had a statistically significant reduction in AIC compared to the intercept-only model.
#  Step 3: Fit every possible two-predictor regression model. Identify the model that produced the lowest AIC and also had a statistically significant reduction in AIC compared to the one-predictor model.
#  Repeat the process until fitting a regression model with more predictor variables no longer leads to a statistically significant reduction in AIC.

#dataset
mtcars

intercept_only <- lm(mpg ~ 1,data = mtcars)
all <- lm(mpg ~ .,data = mtcars)
forward <- step(intercept_only,direction = 'forward',scope = formula(all), trace=0)
forward$anova

#The ANOVA shows that wt, cyl, and hp reduce the AIC (improve the model)
