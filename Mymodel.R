
library(Rcpp)
library(RcppArmadillo)
Rcpp::sourceCpp("./Mymodel/Specific.cpp",verbose = TRUE, rebuild = TRUE)
#Rcpp::sourceCpp("./Mymodel/Share.cpp",verbose = TRUE, rebuild = TRUE)

  # 列表/数据框转数值向量 y1,y2 
  # 基因矩阵转换 m1,m2
  g=ncol(m1)
  n1=length(y1)
  n2=length(y2)
  H1 =  Mymodel_individualcppSpecific(y1=y1,y2=y2,m1=m1,m2=m2,
                              tol=1e-5,maxIter=5000)
 
  loglik0 <- H1$nullloglik_max
  MA1lkvec <- H1$loglkmaxs
  LRT=c()
  
  for(k in 1:g){
    LRT[k]=2*(loglik0 - MA1lkvec[k])
    P[k]=0.5*pchisq(LRT[k], 1, lower.tail=FALSE)+0.25*pchisq(LRT[k], 2, lower.tail=FALSE)
  }
  
listH=list(,loglik0,MA1lkvec,P)
#each gene's ancestry-specific p value is in P
