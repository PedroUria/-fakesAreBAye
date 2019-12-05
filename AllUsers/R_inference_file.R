library(e1071)
library(caret)
install.packages('purrr')
USE_MODE = TRUE
USE_MAJORITY_VOTE = TRUE
set.seed(12)
intercept = TRUE

### read in the data for the coefficients from the train dat
# read in the test data 
# test_data = read.csv(getwd()+'prep_data/test_data.csv')
test_data = read.csv('prep_data/data_test.csv')
test_data$interaction_posR_Review = test_data$avg_posR*test_data$Reviewer_deviation


##### CHANGE FILEPATH TO FILE CONTAINING MOST RECENT BETA CSVS
FILEpath = 'C:/Users/seanp/Documents/Github/-fakesAreBAye/bdata_sean_15'
setwd(FILEpath)
filez = list.files(path = FILEpath)
if(!("beta0.csv" %in% filez)){
  intercept ==FALSE
}
# sorry, but you have to reset your WD to where the bdata is for this to work. 

#tm = read.csv(paste0(FILEpath,filez[1],sep=''),col.names = c('drop',toString(0)))
temp = list.files(pattern="*.csv")
# sorry, I will work out a way that just gets the .csvs out .... my apologies
filez = filez[1:4]
myfiles = lapply(filez,read.csv)

for (i in 1:length(myfiles)){
  myfiles[[i]] = data.frame(myfiles[[i]])
  colnames(myfiles[[i]]) = c('drop',paste0('beta',toString(i-1),sep=''))
  myfiles[[i]] = myfiles[[i]][,-1]
}

mcmcBs  = data.frame(do.call(cbind,myfiles))

round_df <- function(x, digits) {
  # round all numeric variables
  # x: data frame 
  # digits: number of digits to round
  numeric_columns <- sapply(x, mode) == 'numeric'
  x[numeric_columns] <-  round(x[numeric_columns], digits)
  x
}
mcmcBs[,c('X1','X2','X4')] = round_df(mcmcBs[,c('X1','X2','X4')],2)
mcmcBs[,'X3'] = round_df(mcmcBs[,'X3'],3)

#b_init = read.csv(filez[0])
# beta0 = read.csv(getwd()+'bdat/beta0.csv')
# beta1 = read.csv(getwd()+'bdat/beta1.csv')
# beta2 = read.csv(getwd()+'bdat/beta2.csv')
# beta3 = read.csv(getwd()+'bdat/beta3.csv')
# beta4 = read.csv(getwd()+'bdat/beta4.csv')
# beta5 = read.csv(getwd()+'bdat/beta5.csv')
# beta6 = read.csv(getwd()+'bdat/beta6.csv')
# beta7 = read.csv(getwd()+'bdat/beta7.csv')

if (USE_MODE) {
  ##define function to get mode value for coefficients 
  
  # use the function to get the mode value... 
  
  Mode <- function(x, na.rm = FALSE) {
    if(na.rm){
      x = x[!is.na(x)]
      
    }
    
    ux <- unique(x)
    return(ux[which.max(tabulate(match(x, ux)))])
  }
  # sorry, you have to enter this part in yourself :( 
  b0m = Mode(mcmcBs[,1])
  b1m = Mode(mcmcBs[,2])
  b2m = Mode(mcmcBs[,3])
  b3m = Mode(mcmcBs[,4])
  # b4m = Mode(mcmcBs[,5])
  # b5m = Mode(mcmcBs[,6])
  # b6m = Mode(mcmcBs[,7])
  
  
  
  # b0m = Mode(beta0$x)
  # b1m = Mode(beta1$x)
  # b2m = Mode(beta2$x)
  # b3m = Mode(beta3$x)
  # b4m = Mode(beta4$x)
  # b5m = Mode(beta5$x)
  # b6m = Mode(beta6$x)
  # b7m = Mode(beta7$x)
  
  thresh <- function(input){
    if(input>=.3){
      output =1
    }else{
      output = 0
    }
    return(output)
  }
  pwedz = sapply(logMu,thresh)
  
  # get the 'mu' in our glm.. which we need to apply logit function to. 
  #xName = c("Reviewer_deviation",'avg_revL','fBERT0')
  ModMu = b0m + (b1m*test_data$Reviewer_deviation) + (b2m * test_data$avg_revL) +  (b3m * test_data$fBERT0)  
  # apply logit function. 
  logMu = 1 / (1 + exp(-1 * ModMu))
  # get predicted values with binomial
  predz = rbinom(n = length(ModMu),size=1,prob = logMu )
  # evaluate predictions! 
  sum(predz == test_data$Fake)/length(test_data$Fake)
  sum(pwedz == test_data$Fake)/length(test_data$Fake)
  
  # confusion matrix
  print(confusionMatrix(factor(pwedz), factor(test_data$Fake), mode = "prec_recall", positive="1"))
  print('above me is using thresholding with mode, below me is USING BINOMIAL WITH MODE')
  confusionMatrix(factor(predz), factor(test_data$Fake), mode = "prec_recall", positive="1")
  
}


#tito = data.frame(mapply(`*`,mcmcBs[,2:6],test_data[1,2:6]))
#t2 = data.frame(mapply(`*`,mcmcBs[,2:8],test_data[1,2:8]))
#tito + mcmcBs[1,1]




if (USE_MAJORITY_VOTE) {
  set.seed(12)
  tF <- function(input,x, intercept = TRUE){
    if(intercept == TRUE){
      output = data.frame(mapply(`*`,input[2:length(input)],  x[,2:length(x)]))
      ModMu = rowSums(output) + x[,1]
      logMu = 1 / (1 + exp(-1 * ModMu))
      output = logMu
      
      
      
      
    }
    if(intercept == FALSE){
      output = data.frame(mapply(`*`,input[1:length(input)],  x[,1:length(x)]))
      ModMu = rowSums(output)
      logMu = 1 / (1 + exp(-1 * ModMu))
    }
    return(output)
  }
  new_test = test_data[,1:length(myfiles)]
  predz = apply(new_test,1,tF,x=mcmcBs)
  
  m_thresh <- function(input){
    if(mean(input)>=.1){
      output =1
    }else{
      output = 0
    }
    return(output)
  }
  maj_preds = apply(predz,2,m_thresh)
  
  
  sum(maj_preds == test_data$Fake)/length(test_data$Fake)
  
  
  
}


### so this is me toying around with the idea of doing a bi old majority vote for everything 
confusionMatrix(factor(maj_preds,levels = c(0,1)), factor(test_data$Fake), mode = "prec_recall", positive="1")