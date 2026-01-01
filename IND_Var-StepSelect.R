# Ref:  https://www.statology.org/stepwise-regression-r/

install.packages("ISLR")
library(ISLR)

#dataset
Hitters

fit = lm(Salary ~ .,data = Hitters)
fit_aic_back = step(fit,trace = FALSE)
summary(fit_aic_back)
# Elements with 3 * are best steps (smallest P values)

coef(fit_aic_back)

