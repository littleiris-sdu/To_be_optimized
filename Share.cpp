
#define ARMA_64BIT_WORD 
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <R.h>
#include <Rmath.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h> 
#include <cstring>
#include <ctime>
#include <Rcpp.h>
#include <omp.h>

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace arma;
using namespace std;

#include <armadillo>

//已知M1和M2，构造diag(M1,M2)的分块对角矩阵函数： 
// 分块对角矩阵构造函数 , 优化内存分配
arma::mat construct_block_diag(const arma::mat& M1, const arma::mat& M2) {
    int n1 = M1.n_rows, n2 = M2.n_rows;
    int p1 = M1.n_cols, p2 = M2.n_cols;
    arma::mat M(n1 + n2, p1 + p2, fill::zeros);
    M.submat(0,  0, n1-1, p1-1) = M1;
    M.submat(n1,  p1, n1+n2-1, p1+p2-1) = M2;
    return M;
}
 
arma::mat construct_block(const arma::mat& A, const arma::mat& B, const arma::mat& C, const arma::mat& D){
    // 创建零矩阵块 
    //水平拼接
    arma::mat Upper = arma::join_horiz(A, B);  // [A | B]
    arma::mat Lower = arma::join_horiz(C, D);  // [C | D]
    //垂直拼接
    arma::mat M = arma::join_vert(Upper, Lower);  // [A B; C D]
    //纵向拼接
    return M;
}

// [[Rcpp::export]]  
List Mymodel_PXEM_NULL_individualcpp(
    const arma::vec Y1, 
	const arma::vec Y2, 
	const arma::mat M1, 
	const arma::mat M2, 
	const int n1,
	const int n2,
	const double tol,
    const int maxIter){     
 // const std::string mode,        // 模式（null/alternative）
 // const bool verbose             // 调试输出 
	int g = M1.n_cols;
	int N = n1+n2;
	//参数初始化 
	arma::vec D1 = (0.01/g) * arma::ones<arma::vec>(g);
    arma::vec D2 = (0.01/g) * arma::ones<arma::vec>(g);
	arma::vec R = arma::zeros<arma::vec>(g);
	double lambda = 1.0;        // lambda扩展参数初始值-固定
	double sigma2y1 = 1.0; // sd1方差
	double sigma2y2 = 1.0; // sd2方差初
	double sigma2y1_inv = 1.0; 
	double sigma2y2_inv = 1.0; 
	 
    //循环中使用的中间变量 
    arma::mat M1TM1 = M1.t() * M1; 
    arma::mat M2TM2 = M2.t() * M2; 
    arma::mat M1TY1 = M1.t() * Y1; 
    arma::mat M2TY2 = M2.t() * Y2; 
    double Y1TY1 = arma::as_scalar(Y1.t() *Y1);
	double Y2TY2 = arma::as_scalar(Y2.t() *Y2);
	//arma::vec Y = arma::join_cols(Y1, Y2); 
	//arma::mat M = construct_block_diag(M1,M2);
	arma::vec Y1_temp;
	arma::vec Y2_temp;
	arma::mat SigmaA_inv_1,SigmaA_inv_4;
	
	double Tr1, Tr2, l1, l2, l3, l4, fenmu2, fenmu1, fenzi;
	
	//似然相关变量 
	arma::vec loglik(maxIter, fill::zeros);
    int final_iter; // 记录实际迭代次数
	
    double logdet1, sign1,lambda2;                       // 行列式计算辅助变量 
	arma::mat SigmaA;
	arma::vec muA1,muA2;
	
	
	//主循环
	for (int iter = 1; iter < maxIter; iter ++ ){
      // E-step 
      // 1. 更新后验协方差矩阵SigmaA(分块协方差阵) 
      //------------------
      SigmaA_inv_1 = sigma2y1_inv * M1TM1;
      SigmaA_inv_4 = sigma2y2_inv * M2TM2;
      arma::mat SigmaA_inv_2 = arma::zeros<arma::mat>(g,g);


      double log_Sigmaa =0; 
      
        for (int i = 0; i < g; i++) {
        	
         double kappa = D1(i) * D2(i) - R(i) * R(i);
         //double kappa = std::max(D1(i)*D2(i) - R(i)*R(i), 1e-12);
		  
            if (kappa < 1e-15) {
            //throw std::runtime_error("*****kappa near 0, cant inv*****");
            cout << "*****kappa near 0!!!!!!!!!!!!!! **" << endl;
            cout << "*****kappa=" << kappa << endl;
            cout << "*****D1=" << D1(i) << endl;
            cout << "*****D2=" << D2(i) << endl;
            cout << "*****R=" << R(i) << endl;
            //kappa = kappa + 1e-12;
            //double kappa = std::max(D1(i)*D2(i) - R(i)*R(i), 1e-12); 
            }
            
        log_Sigmaa += std::log(kappa);     
        SigmaA_inv_2.diag()(i) = -R(i) / kappa;
        SigmaA_inv_1(i, i) += D2(i) / kappa;
        SigmaA_inv_4(i, i) += D1(i) / kappa;
        
        }
      
       arma::mat SigmaA_inv = construct_block(SigmaA_inv_1, SigmaA_inv_2, 
                                       SigmaA_inv_2.t(), SigmaA_inv_4);
      
       if(SigmaA_inv.is_sympd()){
         SigmaA = arma::inv_sympd(SigmaA_inv);  // 对称正定矩阵专用 
         } else {
        cout << "***** SigmaA PINV *********************************************" << endl;
        SigmaA = arma::inv(SigmaA_inv, arma::inv_opts::allow_approx);  // 伪逆处理病态情况 arma::inv(SigmaA_inv, arma::inv_opts::allow_approx)
        } 
        
      arma::mat SigmaA_1 = SigmaA.submat(0,  0, g-1, g-1);    // 左上块
      arma::mat SigmaA_2 = SigmaA.submat(0,  g, g-1, 2*g-1);  // 右上块
      arma::mat SigmaA_3 = SigmaA_2.t();                      // 左下块
      arma::mat SigmaA_4 = SigmaA.submat(g,  g, 2*g-1, 2*g-1);// 右下块 
      
 	  // 2. 更新后验均值muA(分块均值) 
	  muA1 = sigma2y1_inv * SigmaA_1 * M1TY1 + sigma2y2_inv * SigmaA_2 * M2TY2; 
	  muA2 = sigma2y1_inv * SigmaA_3 * M1TY1 + sigma2y2_inv * SigmaA_4 * M2TY2;  
	  
	  //计算 log(观测似然)
	  arma::log_det(logdet1, sign1, SigmaA);
	  
	  l1= arma::as_scalar(muA1.t() * SigmaA_inv_1 * muA1);
	  l2= arma::as_scalar(muA2.t() * SigmaA_inv_2.t() * muA1);
	  l3= arma::as_scalar(muA1.t() * SigmaA_inv_2 * muA2);
	  l4= arma::as_scalar(muA2.t() * SigmaA_inv_4 * muA2);
	  
	  loglik(iter) = (  
	   -0.5 * N * std::log(2.0 * M_PI)
	   +0.5 * logdet1
	   -0.5 * log_Sigmaa
	   -0.5 * (n1 * std::log(sigma2y1) + n2 * std::log(sigma2y2) ) 
	   -0.5 * (sigma2y1_inv * Y1TY1 + sigma2y2_inv *Y2TY2)
	   +0.5 * (l1+l2+l3+l4)
       );
       
      cout << "***** doing *****:" <<  iter << endl;
       
      if ( iter>2 && loglik(iter) - loglik(iter - 1) < 0 ){
      perror("Warning: The likelihood failed to increase!");
      }

      if (iter>2 && abs(loglik(iter) - loglik(iter - 1)) < tol) {
      final_iter = iter;
      break;
     }
	  
 // M-step
	  // 1. D1、D2、R的更新Sigmaa
	  for(int i=0; i<g; i++){
	  	    D1(i) = muA1(i) * muA1(i) + SigmaA_1(i, i);
            D2(i) = muA2(i) * muA2(i) + SigmaA_4(i, i);
            R(i)  = muA1(i) * muA2(i) + SigmaA_2(i, i);
	  } 
	
	  // 3. lambda的更新 
	  fenzi = arma::as_scalar(sigma2y1_inv * Y1.t() * M1 * muA1 + sigma2y2_inv * Y2.t() * M2 * muA2);
	  fenmu1 = sigma2y1_inv * arma::as_scalar( muA1.t() * M1TM1 * muA1 ) + sigma2y2_inv * arma::as_scalar(muA2.t() * M2TM2 * muA2);
	  fenmu2 = sigma2y1_inv * arma::trace(M1TM1 * SigmaA_1) + sigma2y2_inv * arma::trace(M2TM2 * SigmaA_4);
	  lambda = fenzi / (fenmu1+fenmu2);
	  lambda2 = lambda*lambda;
	  
	  // 2. Sigmay的更新 
	  Y1_temp = Y1-lambda*M1*muA1;
	  Y2_temp = Y2-lambda*M2*muA2;
	  
	  Tr1 = arma::trace(M1TM1 * SigmaA_1 );
	  Tr2 = arma::trace(M2TM2 * SigmaA_4 );
	  sigma2y1 = (1.0/n1) * ( arma::dot(Y1_temp, Y1_temp) + lambda2*Tr1);
	  sigma2y2 = (1.0/n2) * ( arma::dot(Y2_temp, Y2_temp) + lambda2*Tr2);
	  
	  sigma2y1_inv = 1.0/sigma2y1;
	  sigma2y2_inv = 1.0/sigma2y2;
	  
    // Reduction Step 
    D1 *= lambda2;
    D2 *= lambda2;
    R *= lambda2;
    lambda = 1.0;
    }
      cout << "***** done *****"  << endl;
	  arma::vec loglik_out;
	  loglik_out= loglik.subvec(1,final_iter);
      double loglik_max = loglik(final_iter);
 
      Rcpp::List res = Rcpp::List::create(         // 封装结果列表 
    _["muA1"] = muA1,
    _["muA2"] = muA2,
    _["loglik"] = loglik_out,
    _["loglik_max"] = loglik_max,
    _["D1"] = D1,
    _["D2"] = D2,
    _["R"] = R,
    _["SigmaA_pos"] = SigmaA
  );
  return res;
}//end func

// [[Rcpp::export]]
List Mymodel_individualcppShare(
    const arma::vec y1, 
    const arma::vec y2, 
    const arma::mat m1, 
    const arma::mat m2,
    const int maxIter,             // 最大迭代 
    const double tol) {            // 收敛容忍度

    Rcpp::List nullRes, Res, output;
    int gNULL = m1.n_cols;
    
    cout<< gNULL << endl; 
    cout<< m1.n_cols<< endl; 
    cout<< m2.n_cols<< endl; 
    
    arma::vec loglkmaxs = arma::zeros<arma::vec>(maxIter);
    int N1 = y1.n_elem;
    int N2 = y2.n_elem;
    cout<< N1<< endl; 
     

    // 运行零模型（完整模型）
    nullRes = Mymodel_PXEM_NULL_individualcpp(y1, y2, m1, m2, N1, N2,tol, maxIter);
    cout<< N2<< endl; 
     
    arma::vec D1 = Rcpp::as<arma::vec>(nullRes["D1"]);  
    arma::vec D2 = Rcpp::as<arma::vec>(nullRes["D2"]);  
    arma::vec R = Rcpp::as<arma::vec>(nullRes["R"]); 
	arma::vec muA1 = Rcpp::as<arma::vec>(nullRes["muA1"]);
	arma::vec muA2 = Rcpp::as<arma::vec>(nullRes["muA2"]);
    
    double nullloglik_max = Rcpp::as<double>(nullRes["loglik_max"]);  
    cout<< nullloglik_max << endl; 
    
     

    // 创建原始矩阵的副本，避免修改原始数据
    arma::mat mm1_base = m1;
    arma::mat mm2_base = m2;

    // 逐个基因检验（留一法）
    for (int f = 0; f < gNULL; f++) {
    	
    	                   cout<<"f=" << f << endl; 

        // 从副本创建当前迭代的矩阵
        arma::mat mm1 = mm1_base;
        arma::mat mm2 = mm2_base;

        // 删除第f列
        if (f < mm1.n_cols) {
            mm1.shed_col(f); 
        } else {
            Rcpp::stop("Index f=%d is out of bounds for mm1 with n_cols=%d", f, mm1.n_cols);
        }
        if (f < mm2.n_cols) {
            mm2.shed_col(f); 
        } else {
            Rcpp::stop("Index f=%d is out of bounds for mm2 with n_cols=%d", f, mm2.n_cols);
        }

   
        // 调用零模型（此时是删除第f个基因后的模型）
        Res = Mymodel_PXEM_NULL_individualcpp(y1, y2, mm1, mm2, N1, N2,tol, maxIter );
        // 注意：这里应该是当前模型Res的loglik_max，而不是nullRes的
        loglkmaxs(f) = Rcpp::as<double>(Res["loglik_max"]);
    }

    output = Rcpp::List::create(
        Rcpp::Named("D1") = D1, 
        Rcpp::Named("D2") = D2, 
        Rcpp::Named("R") = R, 
        Rcpp::Named("muA1") = muA1, 
        Rcpp::Named("muA2") = muA2, 
        Rcpp::Named("nullloglik_max") = nullloglik_max, 
        Rcpp::Named("loglkmaxs") = loglkmaxs
    );

    Rcpp::Rcout << "***** All done *****" << std::endl;
    return output;
}
