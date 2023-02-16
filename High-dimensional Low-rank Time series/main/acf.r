y <- read.csv(getwd()+'/temp/y.csv',header = FALSE)
acf <- acf(y,lag.max=50,plot=FALSE)
acf <- apply(acf$acf,1,norm,type='2')
write.csv(acf,getwd()+'/temp/acf.csv')

