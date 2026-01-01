# Ref: https://www.statology.org/principal-components-regression-in-r/

install.packages("pls")
library(pls)

#dataset
mtcars

set.seed(1)
model <- pcr(hp ~ mpg + disp + drat + wt + qsec,data = mtcars,scale = TRUE,validation = "CV")
summary(model)

validationplot(model)
#low point in the graph shows the number of components with the lowest error (best accuracy)
