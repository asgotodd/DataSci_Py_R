# Install the packages and load libraries
 install.packages('arules')
 library('arules')
#Read in Groceries data
data(Groceries)
Groceries
Groceries@itemInfo
#mine rules
rules <- apriori(Groceries, parameter=list(support=0.001, confidence=0.5))
#Extract rules with confidence =0.8
subrules <- rules[quality(rules)$confidence > 0.8]
inspect(subrules)
#Extract the top three rules with high lift 
rules_high_lift <- head(sort(rules, by="lift"), 3)
inspect(rules_high_lift)

