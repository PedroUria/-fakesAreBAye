### read in the data for the coefficients from the train dat
# read in the test data 

test_data = read.csv(getwd()+'prep_data/test_data.csv')


beta0 = read.csv(getwd()+'bdat/beta0.csv')
beta1 = read.csv(getwd()+'bdat/beta1.csv')
beta2 = read.csv(getwd()+'bdat/beta2.csv')
beta3 = read.csv(getwd()+'bdat/beta3.csv')
beta4 = read.csv(getwd()+'bdat/beta4.csv')
beta5 = read.csv(getwd()+'bdat/beta5.csv')
beta6 = read.csv(getwd()+'bdat/beta6.csv')
beta7 = read.csv(getwd()+'bdat/beta7.csv')

##define function to get mode value for coefficients 

# use the function to get the mode value... 

Mode <- function(x, na.rm = FALSE) {
  if(na.rm){
    x = x[!is.na(x)]
  }
  
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}

b0m = Mode (beta0)
b1m = Mode(beta1)
b2m = Mode(beta2)
b3m = Mode(beta3)
b4m = Mode(beta4)
b5m = Mode(beta5)
b6m = Mode(beta6)
b7m = Mode(beta7)

# get the 'mu' in our glm.. which we need to apply logit function to. 
ModMu = b0m + (b1m*test_data$Reviewer_deviation) + (b2m * test_data$avg_posR) + (b3m * test_data$avg_revL) + (b4m * test_data$MNR) + (b5m * test_data$fBERT0) + (b6m * test_data$fBERT1) + (b7m * test_data$fBERT2)
# apply logit function. 
logMu = 1 / (1 + exp(-1 * ModMu))
# get predicted values 
predz = rbinom(n = length(ModMu),size=1,prob = logMu )
# evaluate predictions! 
sum(predz == test_data$Fake)/length(test_data$Fake)