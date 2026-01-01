cereal <- read.csv(file = #<LocalPath> "ANOVA_cereals.csv",stringsAsFactors=TRUE)
cereal$shelf1 <- ifelse(cereal$Shelf==1,1,0)
cereal$shelf2 <- ifelse(cereal$Shelf==2,1,0)
mreg4.1 <- lm(Rating ~ Sugars + Fiber + shelf2,data = cereal)
anova(mreg4.1)

mreg4.2 <- lm(Rating ~ shelf1 + shelf2 + Sugars + Fiber,data = cereal)
anova(mreg4.2)
