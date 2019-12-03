# Jags-Ydich-XmetMulti-Mlogistic.R 
# Accompanies the book:
#   Kruschke, J. K. (2015). Doing Bayesian Data Analysis, Second Edition: 
#   A Tutorial with R, JAGS, and Stan. Academic Press / Elsevier.

source("DBDA2E-utilities.R")
SEED1 = 12
SEED2 = 3
SEED3 = 7
xName = c("Reviewer_deviation",'avg_posR','avg_revL','MNR','fBERT0' ,'fBERT1','fBERT2')
# The first of these modes and stds is for the intercept
PRIORS_MODES = c(0, 1, 1, -1, 1, -0.5025951, -0.37428102, 0.35757005)
PRIORS_STDS = c(1/2^2, 1/2^2, 1/2^2, 1/2^2, 1/2^2, 1/2^2, 1/2^2, 1/2^2)
DO_ROBUST = TRUE  # If true, change the beta distrib for the guess below
GUESS_BETA_A = 1  # Guess is a beta
GUESS_BETA_B = 9
GUESS_MULTIPL = 0.2  # Importance of random guess part
DO_VARIABLE_SELECTION = TRUE
# There is a theta for the intercept
DELTAS_THETAS = c(0.5, 0.9, 0.9, 0.9, 0.9, 0.8, 0.1, 0.1)
BETA_DIR_NAME = "bdata_Pedro_02"

#===============================================================================

genMCMC = function( data , xName="x" , yName="Fake" , 
                    numSavedSteps=10000 , thinSteps=1 , saveName=NULL ,
                    runjagsMethod=runjagsMethodDefault , 
                    nChains=nChainsDefault ) { 
  require(runjags)
  #-----------------------------------------------------------------------------
  # THE DATA.
  y = data[,yName]
  x = as.matrix(data[,xName],ncol=length(xName))
  # Do some checking that data make sense:
  if ( any( !is.finite(y) ) ) { stop("All y values must be finite.") }
  if ( any( !is.finite(x) ) ) { stop("All x values must be finite.") }
  cat("\nCORRELATION MATRIX OF PREDICTORS:\n ")
  show( round(cor(x),3) )
  cat("\n")
  flush.console()
  # Specify the data in a list, for later shipment to JAGS:
  dataList = list(
    x = x ,
    y = y ,
    Nx = dim(x)[2],
    Ntotal = dim(x)[1],
    priors_modes = PRIORS_MODES,
    priors_stds = PRIORS_STDS,
    guess_beta_a = GUESS_BETA_A,
    guess_beta_b = GUESS_BETA_B,
    guess_multipl = GUESS_MULTIPL,
    deltas_thetas = DELTAS_THETAS
  )
  #-----------------------------------------------------------------------------
  # THE MODEL.
  if ( DO_ROBUST ) {
    modelString = "
    # Standardize the data:
    data {
    for ( j in 1:Nx ) {
    xm[j]  <- mean(x[,j])
    xsd[j] <-   sd(x[,j])
    # if ( j > 5 ) {  # For some reason this doesn't work so I just invers-standardized BERT's features on features.py
    # xm[j]  <- mean(x[,j]) * 0
    # xsd[j] <-   sd(x[,j]) / sd(x[,j])
    # }
    for ( i in 1:Ntotal ) {
    zx[i,j] <- ( x[i,j] - xm[j] ) / xsd[j]
    }
    }
    }
    # Specify the model for standardized data:
    model {
    for ( i in 1:Ntotal ) {
    # In JAGS, ilogit is logistic:
    y[i] ~ dbern( guess*guess_multipl + (1-guess)*ilogit( zbeta0 + sum( zbeta[1:Nx] * zx[i,1:Nx] ) ) )
    }
    # Priors:
    zbeta0 ~ dnorm( priors_modes[1] , priors_stds[1] )  # TODO: Change to some kind of beta
    # for (i in 1:Nx + 1) {
    # zbeta[i] ~ dnorm( priors_modes[i] , 1/2^2 )
    # }
    zbeta[1] ~ dnorm( priors_modes[2] , priors_stds[2] )  # Reviwer deviation, greater means more likely to be fake
    zbeta[2] ~ dnorm( priors_modes[3] , priors_stds[3])  # Percentage of 4-5 star reviews per user, greater means more likely to be fake
    zbeta[3] ~ dnorm( priors_modes[4] , priors_stds[4] )  # For avg review length per user, longer reviews means more likely to be real
    zbeta[4] ~ dnorm( priors_modes[5] , priors_stds[5] )  # Maximum number of reviews in a day per user, more reviews means more likely to be fake
    zbeta[5] ~ dnorm( priors_modes[6] , priors_stds[6] )  # BERT features
    zbeta[6] ~ dnorm( priors_modes[7] , priors_stds[7] )  # using weights
    zbeta[7] ~ dnorm( priors_modes[8] , priors_stds[8] )  # as mode for priors
    guess ~ dbeta(guess_beta_a, guess_beta_b)
    # for ( j in 1:Nx ) {
    # zbeta[j] ~ dnorm( 0 , 1/2^2 )
    # }
    # Transform to original scale:
    beta[1:Nx] <- zbeta[1:Nx] / xsd[1:Nx] 
    beta0 <- zbeta0 - sum( zbeta[1:Nx] * xm[1:Nx] / xsd[1:Nx] )
    }
    "
  }
  if (DO_ROBUST == FALSE & DO_VARIABLE_SELECTION == FALSE) {
    modelString = "
    # Standardize the data:
    data {
    for ( j in 1:Nx ) {
    xm[j]  <- mean(x[,j])
    xsd[j] <-   sd(x[,j])
    # if ( j > 5 ) {  # For some reason this doesn't work so I just invers-standardized BERT's features on features.py
    # xm[j]  <- mean(x[,j]) * 0
    # xsd[j] <-   sd(x[,j]) / sd(x[,j])
    # }
    for ( i in 1:Ntotal ) {
    zx[i,j] <- ( x[i,j] - xm[j] ) / xsd[j]
    }
    }
    }
    # Specify the model for standardized data:
    model {
    for ( i in 1:Ntotal ) {
    # In JAGS, ilogit is logistic:
    y[i] ~ dbern( ilogit( zbeta0 + sum( zbeta[1:Nx] * zx[i,1:Nx] ) ) )
    }
    # Priors
    zbeta0 ~ dnorm( priors_modes[1] , priors_stds[1] )  # TODO: Change to some kind of beta
    # for (i in 1:Nx + 1) {
    # zbeta[i] ~ dnorm( priors_modes[i] , 1/2^2 )
    # }
    zbeta[1] ~ dnorm( priors_modes[2] , priors_stds[2] )  # Reviwer deviation, greater means more likely to be fake
    zbeta[2] ~ dnorm( priors_modes[3] , priors_stds[3])  # Percentage of 4-5 star reviews per user, greater means more likely to be fake
    zbeta[3] ~ dnorm( priors_modes[4] , priors_stds[4] )  # For avg review length per user, longer reviews means more likely to be real
    zbeta[4] ~ dnorm( priors_modes[5] , priors_stds[5] )  # Maximum number of reviews in a day per user, more reviews means more likely to be fake
    zbeta[5] ~ dnorm( priors_modes[6] , priors_stds[6] )  # BERT features
    zbeta[6] ~ dnorm( priors_modes[7] , priors_stds[7] )  # using weights
    zbeta[7] ~ dnorm( priors_modes[8] , priors_stds[8] )  # as mode for priors
    # Transform to original scale:
    beta[1:Nx] <- zbeta[1:Nx] / xsd[1:Nx] 
    beta0 <- zbeta0 - sum( zbeta[1:Nx] * xm[1:Nx] / xsd[1:Nx] )
    }
    "
  }
  if (DO_VARIABLE_SELECTION) {
    modelString = "
    # Standardize the data:
    data {
    for ( j in 1:Nx ) {
    xm[j]  <- mean(x[,j])
    xsd[j] <-   sd(x[,j])
    # if ( j > 5 ) {  # For some reason this doesn't work so I just invers-standardized BERT's features on features.py
    # xm[j]  <- mean(x[,j]) * 0
    # xsd[j] <-   sd(x[,j]) / sd(x[,j])
    # }
    for ( i in 1:Ntotal ) {
    zx[i,j] <- ( x[i,j] - xm[j] ) / xsd[j]
    }
    }
    }
    # Specify the model for standardized data:
    model {
    for ( i in 1:Ntotal ) {
    # In JAGS, ilogit is logistic:
    y[i] ~ dbern( ilogit( delta0 * zbeta0 + sum( delta[1:Nx] * zbeta[1:Nx] * zx[i,1:Nx] ) ) )
    }
    # Priors
    zbeta0 ~ dnorm( priors_modes[1] , priors_stds[1] )  # TODO: Change to some kind of beta
    # for (i in 1:Nx + 1) {
    # zbeta[i] ~ dnorm( priors_modes[i] , 1/2^2 )
    # }
    zbeta[1] ~ dnorm( priors_modes[2] , priors_stds[2] )  # Reviwer deviation, greater means more likely to be fake
    zbeta[2] ~ dnorm( priors_modes[3] , priors_stds[3])  # Percentage of 4-5 star reviews per user, greater means more likely to be fake
    zbeta[3] ~ dnorm( priors_modes[4] , priors_stds[4] )  # For avg review length per user, longer reviews means more likely to be real
    zbeta[4] ~ dnorm( priors_modes[5] , priors_stds[5] )  # Maximum number of reviews in a day per user, more reviews means more likely to be fake
    zbeta[5] ~ dnorm( priors_modes[6] , priors_stds[6] )  # BERT features
    zbeta[6] ~ dnorm( priors_modes[7] , priors_stds[7] )  # using weights
    zbeta[7] ~ dnorm( priors_modes[8] , priors_stds[8] )  # as mode for priors
    delta0 ~ dbern ( deltas_thetas[1] )
    delta[1] ~ dbern ( deltas_thetas[2] )
    delta[2] ~ dbern ( deltas_thetas[3] )
    delta[3] ~ dbern ( deltas_thetas[4] )
    delta[4] ~ dbern ( deltas_thetas[5] )
    delta[5] ~ dbern ( deltas_thetas[6] )
    delta[6] ~ dbern ( deltas_thetas[7] )
    delta[7] ~ dbern ( deltas_thetas[8] )
    guess ~ dbeta(guess_beta_a, guess_beta_b)
    # Transform to original scale:
    beta[1:Nx] <- zbeta[1:Nx] / xsd[1:Nx] 
    beta0 <- zbeta0 - sum( zbeta[1:Nx] * xm[1:Nx] / xsd[1:Nx] )
    }
    "
  }
  if (DO_ROBUST & DO_VARIABLE_SELECTION) {
    modelString = "
    # Standardize the data:
    data {
    for ( j in 1:Nx ) {
    xm[j]  <- mean(x[,j])
    xsd[j] <-   sd(x[,j])
    # if ( j > 5 ) {  # For some reason this doesn't work so I just invers-standardized BERT's features on features.py
    # xm[j]  <- mean(x[,j]) * 0
    # xsd[j] <-   sd(x[,j]) / sd(x[,j])
    # }
    for ( i in 1:Ntotal ) {
    zx[i,j] <- ( x[i,j] - xm[j] ) / xsd[j]
    }
    }
    }
    # Specify the model for standardized data:
    model {
    for ( i in 1:Ntotal ) {
    # In JAGS, ilogit is logistic:
    y[i] ~ dbern(  guess*guess_multipl + (1-guess)*ilogit( delta0 * zbeta0 + sum( delta[1:Nx] * zbeta[1:Nx] * zx[i,1:Nx] ) ) )
    }
    # Priors
    zbeta0 ~ dnorm( priors_modes[1] , priors_stds[1] )  # TODO: Change to some kind of beta
    # for (i in 1:Nx + 1) {
    # zbeta[i] ~ dnorm( priors_modes[i] , 1/2^2 )
    # }
    zbeta[1] ~ dnorm( priors_modes[2] , priors_stds[2] )  # Reviwer deviation, greater means more likely to be fake
    zbeta[2] ~ dnorm( priors_modes[3] , priors_stds[3])  # Percentage of 4-5 star reviews per user, greater means more likely to be fake
    zbeta[3] ~ dnorm( priors_modes[4] , priors_stds[4] )  # For avg review length per user, longer reviews means more likely to be real
    zbeta[4] ~ dnorm( priors_modes[5] , priors_stds[5] )  # Maximum number of reviews in a day per user, more reviews means more likely to be fake
    zbeta[5] ~ dnorm( priors_modes[6] , priors_stds[6] )  # BERT features
    zbeta[6] ~ dnorm( priors_modes[7] , priors_stds[7] )  # using weights
    zbeta[7] ~ dnorm( priors_modes[8] , priors_stds[8] )  # as mode for priors
    delta0 ~ dbern ( deltas_thetas[1] )
    delta[1] ~ dbern ( deltas_thetas[2] )
    delta[2] ~ dbern ( deltas_thetas[3] )
    delta[3] ~ dbern ( deltas_thetas[4] )
    delta[4] ~ dbern ( deltas_thetas[5] )
    delta[5] ~ dbern ( deltas_thetas[6] )
    delta[6] ~ dbern ( deltas_thetas[7] )
    delta[7] ~ dbern ( deltas_thetas[8] )
    guess ~ dbeta(guess_beta_a, guess_beta_b)
    # Transform to original scale:
    beta[1:Nx] <- zbeta[1:Nx] / xsd[1:Nx] 
    beta0 <- zbeta0 - sum( zbeta[1:Nx] * xm[1:Nx] / xsd[1:Nx] )
    }
    "
  }
  # close quote for modelString
  # Write out modelString to a text file
  writeLines( modelString , con="TEMPmodel.txt" )
  
  #-----------------------------------------------------------------------------
  # INTIALIZE THE CHAINS
  # https://sourceforge.net/p/mcmc-jags/discussion/610037/thread/359a4c90/?limit=25
  inits1 <- list(m=1, c=1, precision=1,
                 .RNG.name="base::Super-Duper", .RNG.seed=SEED1)
  inits2 <- list(m=0.1, c=10, precision=1,
                 .RNG.name="base::Wichmann-Hill", .RNG.seed=SEED2)
  inits3 <- list(m=0.2, c=20, precision=1,
                 .RNG.name="base::Marsaglia-Multicarry", .RNG.seed=SEED3)
  
  #-----------------------------------------------------------------------------
  # RUN THE CHAINS
  parameters = c( "beta0" ,  "beta" ,  
                  "zbeta0" , "zbeta", "delta0", "delta", "guess" )
  adaptSteps = 500  # Number of steps to "tune" the samplers
  burnInSteps = 1000
  runJagsOut <- run.jags( method=runjagsMethod ,
                          model="TEMPmodel.txt" , 
                          monitor=parameters , 
                          data=dataList ,  
                          inits=list(inits1,inits2,inits3) , 
                          n.chains=nChains ,
                          adapt=adaptSteps ,
                          burnin=burnInSteps , 
                          sample=ceiling(numSavedSteps/nChains) ,
                          thin=thinSteps ,
                          summarise=FALSE ,
                          plots=FALSE )
  codaSamples = as.mcmc.list( runJagsOut )
  # resulting codaSamples object has these indices: 
  #   codaSamples[[ chainIdx ]][ stepIdx , paramIdx ]
  if ( !is.null(saveName) ) {
    save( codaSamples , file=paste(saveName,"Mcmc.Rdata",sep="") )
  }
  return( codaSamples )
} # end function

#===============================================================================

smryMCMC = function(  codaSamples , 
                      saveName=NULL ) {
  summaryInfo = NULL
  mcmcMat = as.matrix(codaSamples)
  paramName = colnames(mcmcMat)
  for ( pName in paramName ) {
    summaryInfo = rbind( summaryInfo , summarizePost( mcmcMat[,pName] ) )
  }
  rownames(summaryInfo) = paramName
  if ( !is.null(saveName) ) {
    write.csv( summaryInfo , file=paste(saveName,"SummaryInfo.csv",sep="") )
  }
  return( summaryInfo )
}

#===============================================================================

plotMCMC = function( codaSamples , data , xName="x" , yName="y" ,
                     showCurve=FALSE ,  pairsPlot=FALSE ,
                     saveName=NULL , saveType="jpg" ) {
  # showCurve is TRUE or FALSE and indicates whether the posterior should
  #   be displayed as a histogram (by default) or by an approximate curve.
  # pairsPlot is TRUE or FALSE and indicates whether scatterplots of pairs
  #   of parameters should be displayed.
  #-----------------------------------------------------------------------------
  y = data[,yName]
  x = as.matrix(data[,xName])
  mcmcMat = as.matrix(codaSamples,chains=TRUE)
  chainLength = NROW( mcmcMat )
  zbeta0 = mcmcMat[,"zbeta0"]
  zbeta  = mcmcMat[,grep("^zbeta$|^zbeta\\[",colnames(mcmcMat))]
  # if (DO_ROBUST) {
  #   guess = mcmcMat[,"guess"]
  # }
  # if (DO_VARIABLE_SELECTION) {
  #   delta = mcmcMat[,grep("^delta$|^delta\\[",colnames(mcmcMat))]
  # }
  if ( ncol(x)==1 ) { zbeta = matrix( zbeta , ncol=1 ) }
  beta0 = mcmcMat[,"beta0"]
  beta  = mcmcMat[,grep("^beta$|^beta\\[",colnames(mcmcMat))]
  if ( ncol(x)==1 ) { beta = matrix( beta , ncol=1 ) }
  # if ( ncol(x)==1 ) { delta = matrix( delta , ncol=1 ) }
  #-----------------------------------------------------------------------------
  if ( pairsPlot ) {
    # Plot the parameters pairwise, to see correlations:
    openGraph()
    nPtToPlot = 1000
    plotIdx = floor(seq(1,chainLength,by=chainLength/nPtToPlot))
    panel.cor = function(x, y, digits=2, prefix="", cex.cor, ...) {
      usr = par("usr"); on.exit(par(usr))
      par(usr = c(0, 1, 0, 1))
      r = (cor(x, y))
      txt = format(c(r, 0.123456789), digits=digits)[1]
      txt = paste(prefix, txt, sep="")
      if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
      text(0.5, 0.5, txt, cex=1.5 ) # was cex=cex.cor*r
    }
    pairs( cbind( beta0 , beta )[plotIdx,] ,
           labels=c( "beta[0]" , 
                     paste0("beta[",1:ncol(beta),"]\n",xName) ) , 
           lower.panel=panel.cor , col="skyblue" )
    if ( !is.null(saveName) ) {
      saveGraph( file=paste(saveName,"PostPairs",sep=""), type=saveType)
    }
  }
  #-----------------------------------------------------------------------------
  # Data with posterior predictive:
  # If only 1 predictor:
  if ( ncol(x)==1 ) {
    openGraph(width=7,height=6)
    par( mar=c(3.5,3.5,2,1) , mgp=c(2.0,0.7,0) )
    plot( x[,1] , y , xlab=xName[1] , ylab=yName , 
          cex=2.0 , cex.lab=1.5 , col="black" , main="Data with Post. Pred." )
    abline(h=0.5,lty="dotted")
    cVec = floor(seq(1,chainLength,length=30))
    xWid=max(x)-min(x)
    xComb = seq(min(x)-0.1*xWid,max(x)+0.1*xWid,length=201)
    for ( cIdx in cVec ) {
      lines( xComb , 1/(1+exp(-(beta0[cIdx]+beta[cIdx,1]*xComb ))) , lwd=1.5 ,
             col="skyblue" )
      xInt = -beta0[cIdx]/beta[cIdx,1]
      arrows( xInt,0.5, xInt,-0.04, length=0.1 , col="skyblue" , lty="dashed" )
    }
    if ( !is.null(saveName) ) {
      saveGraph( file=paste(saveName,"DataThresh",sep=""), type=saveType)
    }
  }
  # If only 2 predictors:
  if ( ncol(x)==2 ) {
    openGraph(width=7,height=7)
    par( mar=c(3.5,3.5,2,1) , mgp=c(2.0,0.7,0) )
    plot( x[,1] , x[,2] , pch=as.character(y) , xlab=xName[1] , ylab=xName[2] ,
          col="black" , main="Data with Post. Pred.")
    cVec = floor(seq(1,chainLength,length=30))
    for ( cIdx in cVec ) {
      abline( -beta0[cIdx]/beta[cIdx,2] , -beta[cIdx,1]/beta[cIdx,2] , col="skyblue" )
    }
    if ( !is.null(saveName) ) {
      saveGraph( file=paste(saveName,"DataThresh",sep=""), type=saveType)
    }
  }
  #-----------------------------------------------------------------------------
  # Marginal histograms:
  
  decideOpenGraph = function( panelCount , saveName , finished=FALSE , 
                              nRow=1 , nCol=3 ) {
    # If finishing a set:
    if ( finished==TRUE ) {
      if ( !is.null(saveName) ) {
        saveGraph( file=paste0(saveName,ceiling((panelCount-1)/(nRow*nCol))), 
                   type=saveType)
      }
      panelCount = 1 # re-set panelCount
      return(panelCount)
    } else {
      # If this is first panel of a graph:
      if ( ( panelCount %% (nRow*nCol) ) == 1 ) {
        # If previous graph was open, save previous one:
        if ( panelCount>1 & !is.null(saveName) ) {
          saveGraph( file=paste0(saveName,(panelCount%/%(nRow*nCol))), 
                     type=saveType)
        }
        # Open new graph
        openGraph(width=nCol*7.0/3,height=nRow*2.0)
        layout( matrix( 1:(nRow*nCol) , nrow=nRow, byrow=TRUE ) )
        par( mar=c(4,4,2.5,0.5) , mgp=c(2.5,0.7,0) )
      }
      # Increment and return panel count:
      panelCount = panelCount+1
      return(panelCount)
    }
  }
  
  # Original scale:
  panelCount = 1
  panelCount = decideOpenGraph( panelCount , saveName=paste0(saveName,"PostMarg") )
  histInfo = plotPost( beta0 , cex.lab = 1.75 , showCurve=showCurve ,
                       xlab=bquote(beta[0]) , main="Intercept" )
  for ( bIdx in 1:ncol(beta) ) {
    panelCount = decideOpenGraph( panelCount , saveName=paste0(saveName,"PostMarg") )
    histInfo = plotPost( beta[,bIdx] , cex.lab = 1.75 , showCurve=showCurve ,
                         xlab=bquote(beta[.(bIdx)]) , main=xName[bIdx] )
  }
  # for ( bIdx in 1:ncol(delta) ) {
    # panelCount = decideOpenGraph( panelCount , saveName=paste0(saveName,"PostMarg") )
    # histInfo = plotPost( delta[,bIdx] , cex.lab = 1.75 , showCurve=showCurve ,
    #                      xlab=bquote(delta[.(bIdx)]) , main=xName[bIdx+ncol(beta)+1], ylim="delta")
  # }
  # histInfo = plotPost( guess , cex.lab = 1.75 , showCurve=showCurve ,
  #                      xlab=bquote("guess") , main="guess" )
  panelCount = decideOpenGraph( panelCount , finished=TRUE , saveName=paste0(saveName,"PostMarg") )
  
  # Standardized scale:
  panelCount = 1
  panelCount = decideOpenGraph( panelCount , saveName=paste0(saveName,"PostMargZ") )
  histInfo = plotPost( zbeta0 , cex.lab = 1.75 , showCurve=showCurve ,
                       xlab=bquote(z*beta[0]) , main="Intercept" )
  for ( bIdx in 1:ncol(beta) ) {
    panelCount = decideOpenGraph( panelCount , saveName=paste0(saveName,"PostMargZ") )
    histInfo = plotPost( zbeta[,bIdx] , cex.lab = 1.75 , showCurve=showCurve ,
                         xlab=bquote(z*beta[.(bIdx)]) , main=xName[bIdx] )
  }
  panelCount = decideOpenGraph( panelCount , finished=TRUE , saveName=paste0(saveName,"PostMargZ") )
  
  #-----------------------------------------------------------------------------
}
#===============================================================================



myData = read.csv('prep_data/data_train.csv')  # [1:3, ]  # [1:200,]
yName = "Fake"
fileNameRoot = "twest_stuff" 
numSavedSteps=15000 ; thinSteps=2
# #.............................................................................
# # Add some outliers:
# outlierMat = matrix( c(
#   190,74,0,
#   230,73,0,
#   120,59,1,
#   150,58,1 ) , ncol=3 , byrow=TRUE , 
#   dimnames= list( NULL , c("weight","height","male") ) )
# myData = rbind( myData , outlierMat )
#.............................................................................
graphFileType = "eps" 
#------------------------------------------------------------------------------- 
# Load the relevant model into R's working memory:
# source("Jags-Ydich-XmetMulti-Mlogistic.R")
#------------------------------------------------------------------------------- 
# Generate the MCMC chain:
#startTime = proc.time()
if ( dir.exists(file.path(".", BETA_DIR_NAME)) ) {
  print("Directory already exists!")
}
unlink(BETA_DIR_NAME, recursive=TRUE)
dir.create(BETA_DIR_NAME)
    
mcmcCoda = genMCMC( data=myData , xName=xName , yName=yName , 
                    numSavedSteps=numSavedSteps , thinSteps=thinSteps , 
                    saveName=fileNameRoot, nChains=3)
plotMCMC(mcmcCoda, myData, xName=xName, yName=yName, saveName=paste0(BETA_DIR_NAME, "/mcmc_plots"), saveType="jpg")

summaryInfo = smryMCMC(mcmcCoda, saveName = paste0(BETA_DIR_NAME, "/mcmc_out"))

mc = as.matrix(mcmcCoda, chains = TRUE)
if (DO_ROBUST) {
  diagMCMC(codaObject=mcmcCoda, parName="guess", saveName=paste0(BETA_DIR_NAME, "/"))
  write.csv(mc[, "guess"], paste0(BETA_DIR_NAME, "/" , paste0("guess"), ".csv"))
}
for (i in 0:length(xName)+1) {
  if (i == 1) {
    beta_id = paste0("beta", toString(i-1))
  }
  else {
    beta_id = paste0("beta[", toString(i-1), "]")
  }
  diagMCMC(codaObject=mcmcCoda, parName=beta_id, saveName=paste0(BETA_DIR_NAME, "/"))
  write.csv(mc[, beta_id], paste0(BETA_DIR_NAME, "/" , paste0("beta", toString(i - 1)), ".csv"))
  if (DO_VARIABLE_SELECTION) {
    if (i == 1) {
      delta_id = paste0("delta", toString(i-1))
    }
    else {
      delta_id = paste0("delta[", toString(i-1), "]")
    }
    diagMCMC(codaObject=mcmcCoda, parName=delta_id, saveName=paste0(BETA_DIR_NAME, "/"))
    write.csv(mc[, delta_id], paste0(BETA_DIR_NAME, "/" , paste0("delta", toString(i - 1)), ".csv"))
  }
  
}

