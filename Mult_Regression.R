# Install necessary packages (only needed first time)
install.packages("ggplot2")
install.packages("dplyr")
install.packages("broom")
install.packages("ggpubr")

# Load packages into your R environment
library(ggplot2)
library(dplyr)
library(broom)
library(ggpubr)

#Load data into frame
setwd("<Local Path>")
heart_data <- read.csv("Mult_Reg_heart_data.csv",header=TRUE,sep=",")

#Correlation biking + heart disease
cor(heart_data$biking, heart_data$smoking)

#Histogram of heart disease values and Linearity
hist(heart_data$heart.disease)
plot(heart.disease ~ biking, data=heart_data)
plot(heart.disease ~ smoking, data=heart_data)

#Linear Regression
heart_disease_mod <-lm(heart.disease ~ biking + smoking, data = heart_data)
summary(heart_disease_mod)

#Regression Plot
par(mfrow=c(2,2))
plot(heart_disease_mod)
par(mfrow=c(1,1))

#Plot of Reg Model
plotting.data<-expand.grid(
  biking = seq(min(heart_data$biking), max(heart_data$biking), length.out=30),
  smoking=c(min(heart_data$smoking), mean(heart_data$smoking), max(heart_data$smoking)))
#Model Predictions
plotting.data$predicted.y <- predict.lm(heart_disease_mod, newdata=plotting.data)
#Graph Legend
plotting.data$smoking <- round(plotting.data$smoking, digits = 2)
plotting.data$smoking <- as.factor(plotting.data$smoking)

#Plot orgiginal data
heart.plot <- ggplot(heart_data, aes(x=biking, y=heart.disease)) +
  geom_point()
heart.plot
#Add Regression Line
heart.plot <- heart.plot +
  geom_line(data=plotting.data, aes(x=biking, y=predicted.y, color=smoking), size=1.25)
heart.plot

#Publication Graph
heart.plot <-
  heart.plot +
  theme_bw() +
  labs(title = "Rates of heart disease (% of population) \n as a function of biking to work and smoking",
       x = "Biking to work (% of population)",
       y = "Heart disease (% of population)",
       color = "Smoking \n (% of population)")
heart.plot

