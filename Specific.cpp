
#define ARMA_64BIT_WORD 
#include <RcppArmadillo.h>

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


arma::mat construct_block_diag(const arma::mat& M1, const arma::mat& M2) {
    int n1 = M1.n_rows, n2 = M2.n_rows;
    int p1 = M1.n_cols, p2 = M2.n_cols;
    arma::mat M(n1 + n2, p1 + p2, fill::zeros);
    M.submat(0,  0, n1-1, p1-1) = M1;
    M.submat(n1,  p1, n1+n2-1, p1+p2-1) = M2;
    return M;
}
 
arma::mat construct_block(const arma::mat& A, const arma::mat& B, const arma::mat& C, const arma::mat& D){
    arma::mat Upper = arma::join_horiz(A, B);  // [A | B]
    arma::mat Lower = arma::join_horiz(C, D);  // [C | D]
    arma::mat M = arma::join_vert(Upper, Lower);  // [A B; C D]
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
	int g = M1.n_cols;
	int N = n1+n2;
	//参数初始化 
	arma::mat D1 = 0.01/g * arma::eye<arma::mat>(g, g);
	arma::mat D2 = 0.01/g * arma::eye<arma::mat>(g, g);
	arma::mat R = arma::zeros<arma::mat>(g, g);
	double lambda = 1.0;       
	double lambda2 = 1.0;       
	double sigma2y1 = 1.0; // sd1
	double sigma2y2 = 1.0; // sd2
	double sigma2y1_inv = 1.0; 
	double sigma2y2_inv = 1.0; 

    arma::mat M2TM2 = M2.t() * M2; 
    arma::mat M2TY2 = M2.t() * Y2; 

	arma::mat Sigmaa_inv,SigmaA;
	double Tr1, Tr2, l1, l2, l3, l4, fenmu2, fenmu1, fenzi;
	arma::vec zeros_col = arma::zeros<arma::vec>(g); //  0

 
	arma::mat Sigmaa = construct_block(D1, R, R.t(), D2); //Σa 

//	cout << Sigmay_inv.n_cols <<Sigmay_inv.n_rows << endl;
//	cout << Sigmaa.n_cols << Sigmaa.n_rows<< endl;


	int iter = 0;	
	arma::vec loglik(maxIter, fill::zeros);
    double logdet1 = 0;
	double sign1 = 0;                       
	double logdet2 = 0; 
	double sign2 = 0;
    int final_iter;
	
	arma::mat M1TM1 = M1.t() * M1; 
    arma::mat M1TY1 = M1.t() * Y1; 
	 

    double Y1TY1 = arma::as_scalar(Y1.t() *Y1);
	double Y2TY2 = arma::as_scalar(Y2.t() *Y2);
	//arma::vec Y = arma::join_cols(Y1, Y2); 
	//arma::mat M = construct_block_diag(M1,M2);
	arma::vec Y1_temp;
	arma::vec Y2_temp;
	arma::mat SigmaA_inv_1,SigmaA_inv_2,SigmaA_inv_3,SigmaA_inv_4;

	arma::vec muA1,muA2;
	
	
	//Main loop
	for (int iter = 1; iter < maxIter; iter ++ ){
      // E-step 
     
        if(Sigmaa.is_sympd()){
        Sigmaa_inv = arma::inv_sympd(Sigmaa);  
         } else {
        cout << "***** Sigmaa PINV *********************************************" << endl;
        Sigmaa_inv =arma::inv(Sigmaa, arma::inv_opts::allow_approx);  // ::inv(SigmaA_inv, arma::inv_opts::allow_approx)!!!!!!!!!
        } 
        
        
      arma::mat Sigmaa_inv_1 = Sigmaa_inv.submat(0,  0, g-1, g-1);    
      arma::mat Sigmaa_inv_2 = Sigmaa_inv.submat(0,  g, g-1, 2*g-1);  
      arma::mat Sigmaa_inv_3 = Sigmaa_inv_2.t();                      
      arma::mat Sigmaa_inv_4 = Sigmaa_inv.submat(g,  g, 2*g-1, 2*g-1);
          
      arma::mat SigmaA_inv_1 = sigma2y1_inv * M1TM1 + Sigmaa_inv_1;
      arma::mat SigmaA_inv_2 = Sigmaa_inv_2;
      arma::mat SigmaA_inv_3 = Sigmaa_inv_3;
      arma::mat SigmaA_inv_4 = sigma2y2_inv * M2TM2 + Sigmaa_inv_4;
      
      arma::mat SigmaA_inv = construct_block(SigmaA_inv_1, SigmaA_inv_2, SigmaA_inv_2.t(), SigmaA_inv_4);
      
       if(SigmaA_inv.is_sympd()){
         SigmaA = arma::inv_sympd(SigmaA_inv); 
         } else {
        cout << "***** SigmaA PINV *********************************************" << endl;
        SigmaA = arma::inv(SigmaA_inv, arma::inv_opts::allow_approx);  // inv(SigmaA_inv, arma::inv_opts::allow_approx)
        } 
        
      arma::mat SigmaA_1 = SigmaA.submat(0,  0, g-1, g-1);    
      arma::mat SigmaA_2 = SigmaA.submat(0,  g, g-1, 2*g-1); 
      arma::mat SigmaA_3 = SigmaA_2.t();                      
      arma::mat SigmaA_4 = SigmaA.submat(g,  g, 2*g-1, 2*g-1);
      
 	 
	  muA1 = sigma2y1_inv * SigmaA_1 * M1TY1 + sigma2y2_inv * SigmaA_2 * M2TY2; 
	  muA2 = sigma2y1_inv * SigmaA_3 * M1TY1 + sigma2y2_inv * SigmaA_4 * M2TY2;  
	  
	 
	  arma::log_det(logdet1, sign1, SigmaA);
	  arma::log_det(logdet2, sign2, Sigmaa);
	  
	  l1= arma::as_scalar( arma::trace(muA1.t() * SigmaA_inv_1 * muA1) );
	  l2= arma::as_scalar(arma::trace(muA2.t() * SigmaA_inv_2.t() * muA1));
	  l3= arma::as_scalar(arma::trace(muA1.t() * SigmaA_inv_2 * muA2));
	  l4= arma::as_scalar(arma::trace(muA2.t() * SigmaA_inv_4 * muA2));
	  
	  loglik(iter) = (  
	   -0.5 * N * std::log(2.0 * M_PI)
	   +0.5 * logdet1
	   -0.5 * logdet2
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
	  // 1. D1 D2 R
	  for(int i=0; i<g; i++){
	  	    D1.diag()(i) = muA1(i) * muA1(i) + SigmaA_1.diag()(i);
            D2.diag()(i) = muA2(i) * muA2(i) + SigmaA_4.diag()(i);
            R.diag()(i)  = muA1(i) * muA2(i) + SigmaA_2.diag()(i);
	  } 
	    Sigmaa = construct_block(D1, R, R.t(), D2);
	
	  // 3. lambda 
	  fenzi = arma::as_scalar(sigma2y1_inv * Y1.t() * M1 * muA1 + sigma2y2_inv * Y2.t() * M2 * muA2);
	  fenmu1 = sigma2y1_inv * arma::as_scalar( muA1.t() * M1TM1 * muA1 ) + sigma2y2_inv * arma::as_scalar(muA2.t() * M2TM2 * muA2);
	  fenmu2 = sigma2y1_inv * arma::trace(M1TM1 * SigmaA_1) + sigma2y2_inv * arma::trace(M2TM2 * SigmaA_4);
	  lambda = fenzi / (fenmu1+fenmu2);
	  lambda2 = lambda*lambda;
	  
	  // 2. Sigmay
	  Y1_temp = Y1-lambda*M1*muA1;
	  Y2_temp = Y2-lambda*M2*muA2;
	  
	  Tr1 = arma::trace(M1TM1 * SigmaA_1 );
	  Tr2 = arma::trace(M2TM2 * SigmaA_4 );
	  sigma2y1 = (1.0/n1) * ( arma::dot(Y1_temp, Y1_temp) + lambda2*Tr1);
	  sigma2y2 = (1.0/n2) * ( arma::dot(Y2_temp, Y2_temp) + lambda2*Tr2);
	  
	  sigma2y1_inv = 1.0/sigma2y1;
	  sigma2y2_inv = 1.0/sigma2y2;
	  
    // Reduction Step 
    Sigmaa = lambda2 *Sigmaa;
    lambda = 1.0;
    }
      cout << "***** done *****"  << endl;
	  arma::vec loglik_out;
	  loglik_out= loglik.subvec(1,final_iter);
      double loglik_max = loglik(final_iter);
 
      Rcpp::List res = Rcpp::List::create(        
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
List Mymodel_PXEM_SA1_individualcpp(
    const arma::vec Y1, 
	const arma::vec Y2, 
	const arma::mat MM1, 
	const arma::mat M2, 
	const int n1,
	const int n2,
    const int maxIter,  
    const double tol){     


    int gNULL = M2.n_cols; 
    int N = n1+n2;
	arma::vec loglike_max(gNULL);
	double Y2TY2 = arma::as_scalar(Y2.t() * Y2); 
    double Y1TY1 = arma::as_scalar(Y1.t() * Y1); 
	
   for( int f = 0; f < gNULL; f++){
    	
	arma::mat M1= MM1; 			
//	M1.shed_row(f);
	M1.shed_col(f);
	
	int g = M1.n_cols;             // g = gNLL-1 
	
	if(g != gNULL-1){
	cout << "WARNING: g in M1 != gNULL-1 ";
	}
			
			
	
	arma::mat D1 = 0.01/g * arma::eye<arma::mat>(g, g);
	arma::mat D2 = 0.01/gNULL * arma::eye<arma::mat>(gNULL, gNULL);
	arma::mat R = arma::zeros<arma::mat>(g, gNULL);
	double lambda = 1.0;    
	double lambda2 = 1.0;    
	double sigma2y1 = 1.0;  // 
	double sigma2y2 = 1.0;  // 
	double sigma2y1_inv = 1.0;
	double sigma2y2_inv = 1.0;

    arma::mat M2TM2 = M2.t() * M2; 
    arma::mat M2TY2 = M2.t() * Y2; 

	arma::mat R_media,Sigmaa_inv,SigmaA;
	arma::vec mediamat1;
	arma::vec mediamat2;
	double Tr1, Tr2, l1, l2, l3, l4, fenmu2, fenmu1, fenzi;
	arma::vec zeros_col = arma::zeros<arma::vec>(g); // 0 


 
	arma::mat Sigmaa = construct_block(D1, R, R.t(), D2); // Σa 

//	cout << Sigmay_inv.n_cols <<Sigmay_inv.n_rows << endl;
//	cout << Sigmaa.n_cols << Sigmaa.n_rows<< endl;
	
	
	int final_iter;	
	arma::vec loglik = arma::zeros<arma::vec>(maxIter);               
    double logdet1 = 0;
	double sign1 = 0;                       
	double logdet2 = 0; 
	double sign2 = 0;

	
	arma::mat M1TM1 = M1.t() * M1; 
    arma::mat M1TY1 = M1.t() * Y1; 
	
	cout << "Begin: PX-EM ";
	
//	while(iter < 2 || (iter < maxIter && std::abs(loglik(iter - 1) - loglik(iter - 2)) >= tol)){

	for (int iter = 1; iter < maxIter; iter ++ ){
			
	  arma::mat R_media3 = arma::zeros<arma::mat>(g, g);
		
      // E-step 
      // 1. SigmaA
        if(Sigmaa.is_sympd()){
        Sigmaa_inv = arma::inv_sympd(Sigmaa);  // 
         } else {
        cout << "***** Sigmaa PINV *********************************************" << endl;
        Sigmaa_inv =arma::inv(Sigmaa, arma::inv_opts::allow_approx);  // ::inv(SigmaA_inv, arma::inv_opts::allow_approx)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        } 
    
      arma::mat Sigmaa_inv_1 = Sigmaa_inv.submat(0,  0, g-1, g-1);    //
      arma::mat Sigmaa_inv_2 = Sigmaa_inv.submat(0,  g, g-1, g+gNULL-1);  // 
      arma::mat Sigmaa_inv_3 = Sigmaa_inv_2.t();                      // 
      arma::mat Sigmaa_inv_4 = Sigmaa_inv.submat(g,  g, g+gNULL-1, g+gNULL-1);//  
      
      arma::mat SigmaA_inv_1 = sigma2y1_inv * M1TM1 + Sigmaa_inv_1;
      arma::mat SigmaA_inv_2 = Sigmaa_inv_2;
      arma::mat SigmaA_inv_3 = Sigmaa_inv_3;
      arma::mat SigmaA_inv_4 = sigma2y2_inv * M2TM2 + Sigmaa_inv_4;
      
      arma::mat SigmaA_inv = construct_block(SigmaA_inv_1, SigmaA_inv_2, SigmaA_inv_3, SigmaA_inv_4);
 
		if(SigmaA_inv.is_sympd()){
          SigmaA = arma::inv_sympd(SigmaA_inv);  // 
         } else {
        cout << "***** SigmaA_inV PINV *********************************************" << endl;
        SigmaA = arma::inv(SigmaA_inv, arma::inv_opts::allow_approx);  // inv(SigmaA_inv, arma::inv_opts::allow_approx)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        } 
	  
      arma::mat SigmaA_1 = SigmaA.submat(0,  0, g-1, g-1);    //
      arma::mat SigmaA_2 = SigmaA.submat(0,  g, g-1, g+gNULL-1);  // 
      arma::mat SigmaA_3 = SigmaA_2.t();                      // 
      arma::mat SigmaA_4 = SigmaA.submat(g,  g, g+gNULL-1, g+gNULL-1);// 
      
 	  // 2. muA
	  arma::vec muA1 = sigma2y1_inv * SigmaA_1 * M1TY1 + sigma2y2_inv * SigmaA_2 * M2TY2; 
	  arma::vec muA2 = sigma2y1_inv * SigmaA_3 * M1TY1 + sigma2y2_inv * SigmaA_4 * M2TY2; 
	  //arma::vec muA = arma::join_cols(muA1, muA2); 
	  
	
	  arma::log_det(logdet1, sign1, SigmaA);
	  arma::log_det(logdet2, sign2, Sigmaa);
	  
	  l1= arma::as_scalar( arma::trace(muA1.t() * SigmaA_inv_1 * muA1) );
	  l2= arma::as_scalar(arma::trace(muA2.t() * SigmaA_inv_2.t() * muA1));
	  l3= arma::as_scalar(arma::trace(muA1.t() * SigmaA_inv_2 * muA2));
	  l4= arma::as_scalar(arma::trace(muA2.t() * SigmaA_inv_4 * muA2));
	  
	  loglik(iter) =  
	   -0.5 * N * std::log(2.0 * M_PI)
	   +0.5 * logdet1 -0.5 * logdet2
	   -0.5 * (n1 * std::log(sigma2y1) + n2 * std::log(sigma2y2) ) 
	   -0.5 * (sigma2y1_inv * Y1TY1 + sigma2y2_inv *Y2TY2)
	   +0.5 * (l1+l2+l3+l4) ;
       
      cout << "***** doing *****:" <<  iter << endl;
       
      if ( iter>2 && loglik(iter) - loglik(iter - 1) < 0 ){
      perror("Warning: The likelihood failed to increase!");
      }

      if (iter>2 && abs(loglik(iter) - loglik(iter - 1)) < tol) {
      final_iter = iter;
      break;
     }
	  // M-step
	 
	  R_media = muA1*muA2.t() + SigmaA_2;
	  R_media.shed_col(f);  // 删除第i列 
	  
//	  cout << g << R_media.n_cols<<  R_media.n_rows<<"expected = g" << endl;
	  
	  for(int i = 0; i < g; i++){
	  	D1.diag()(i) = muA1(i)*muA1(i) + SigmaA_1.diag()(i);
	  	R_media3.diag()(i) = R_media.diag()(i);
	  } 
	  R_media3.insert_cols(f, zeros_col);
	  R = R_media3;
//	  cout <<  R_media3.n_rows<< R_media3.n_cols<<"expected = g*gNULL" << endl;
	  
//     cout<< "R_media3: " << R_media3.n_rows<< R_media3.n_cols<< "R_media3: expected = g*gNULL" << endl;
//     cout<< "R: " << R.n_cols << R.n_rows<< "R: expected = g*gNULL" << endl;
	  
	  for(int i = 0; i < gNULL; i++){
	  	D2.diag()(i) = muA2(i)*muA2(i) + SigmaA_4.diag()(i);
	  } 
	  
//	  cout<< "D2: " << D2.n_cols << D2.n_rows<< "D2: expected = gNULL*gNULL" << endl;
//	  cout<< "D1: " << D1.n_cols << D1.n_rows<< "D1: expected = g*g" << endl;
//	  cout<< "R: " << R.n_cols << R.n_rows<< "R: expected = g*gNULL" << endl;
	  
	  Sigmaa = construct_block(D1, R, R.t(), D2);
//	  cout << Sigmaa.n_cols << Sigmaa.n_rows<< endl;
	
	  // 2. Sigmay 
	  mediamat1 = Y1-lambda*M1*muA1;
	  mediamat2 = Y2-lambda*M2*muA2;
	  Tr1 = arma::trace(M1 * SigmaA_1 * M1.t());
	  Tr2 = arma::trace(M2 * SigmaA_4 * M2.t());
	  sigma2y1 = (1.0/n1) * ( arma::as_scalar(mediamat1.t() * mediamat1) + lambda*lambda*Tr1);
	  sigma2y2 = (1.0/n2) * ( arma::as_scalar(mediamat2.t() * mediamat2) + lambda*lambda*Tr2);
	  sigma2y1_inv = 1.0 / sigma2y1 ;
	  sigma2y2_inv = 1.0 / sigma2y2 ;	  
	  // 3. lambda 
	  fenzi = arma::as_scalar(sigma2y1_inv * Y1.t() * M1 * muA1 + sigma2y2_inv * Y2.t() * M2 * muA2);
	  fenmu1 = sigma2y1_inv * arma::as_scalar( muA1.t() * M1TM1 * muA1 ) + sigma2y2_inv * arma::as_scalar(muA2.t() * M2TM2 * muA2);
	  fenmu2 = sigma2y1_inv * arma::trace(M1TM1 * SigmaA_1) + sigma2y2_inv * arma::trace(M2TM2 * SigmaA_4);
	  lambda = fenzi / (fenmu1+fenmu2);
	  lambda2 = lambda*lambda;
	  
	  // Reduction Step
	  Sigmaa = lambda2 * Sigmaa;
	  lambda = 1;
	  
    }
    
	  arma::vec loglik_out;
      double loglik_max = loglik(final_iter);
      loglike_max(f) = loglik_max;
 //     cout << "f: "<<f << endl;
 //     cout << "d1 :"<<D1.diag() << endl;
 //     cout << "d2 :"<<D2.diag() << endl;
  //    cout << "R :"<<R.diag() << endl;
	}

    Rcpp::List res = Rcpp::List::create(        
    _["loglike_max"] = loglike_max
  );
  return res;
}//end func



// [[Rcpp::export]]
List Mymodel_individualcppSpecific(
    const arma::vec y1, 
	const arma::vec y2, 
	const arma::mat m1, 
	const arma::mat m2,
	const int maxIter,             // 
    const double tol){     //
	
    Rcpp::List nullRes, SA1Res, SA2Res,output;
    arma::vec SA1_loglkmaxs;
    arma::vec SA2_loglkmaxs;
 
    int N1 = y1.n_elem;
    int N2 = y2.n_elem;
    int gNULL = m1.n_cols;
	 
    if (gNULL != (int)m2.n_cols){
      perror("The dimensions in predicted gene expression of 2 ancestries are not matched in M");
    }
    
    nullRes = Mymodel_PXEM_NULL_individualcpp(y1,y2,m1,m2,N1,N2, tol, maxIter );
	double nullloglik_max = Rcpp::as<double>(nullRes["loglik_max"]);  
	cout << "full model done.";	 
	
	
	 
	SA1Res = Mymodel_PXEM_SA1_individualcpp(y1, y2, m1, m2, N1, N2,  maxIter,tol);
	SA1_loglkmaxs = Rcpp::as<arma::vec>(SA1Res["loglike_max"]); 
	cout << "A1 test done.";	 
	
	SA2Res = Mymodel_PXEM_SA1_individualcpp(y2, y1, m2, m1, N2, N1,  maxIter,tol);
	SA2_loglkmaxs = Rcpp::as<arma::vec>(SA2Res["loglike_max"]); 
	cout << "A2 test done.";	 
	 
  auto end = chrono::steady_clock::now();
  output = Rcpp::List::create(
    _["nullRes"] = nullRes, 
    _["nullloglik_max"] = nullloglik_max, 
    _["SA1_loglkmaxs"] = SA1_loglkmaxs,      // 
    _["SA2_loglkmaxs"] = SA2_loglkmaxs  // P
    / 
  );

  cout << "***** done *****" << endl;
  return output;
}

