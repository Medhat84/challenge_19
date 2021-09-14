library(lightgbm);library(parallel);library(doParallel);library(lubridate);
library(dplyr);library(zoo);library(caret);library(xgboost);library(catboost);
clus <- makeCluster(detectCores()-0);
registerDoParallel(clus);

setwd("~/R/R Directory/challenge_19");
trainds <- read.csv("train.csv"); testds <- read.csv("test.csv");
trainds$datetime <- ymd_h(trainds$date); trainds <- trainds[c(8,2:7)]
set.seed(145);


#inTrain_9 <- 9+which(seq(26159,0) %% 84 > 56);
#inTrain_18 <- 18+which(seq(26159,0) %% 84 > 65);
#inTrain_27 <- 27+which(seq(26159,0) %% 84 > 74);

stdev <- function(x, ...) {x<- x[!is.na(x)];sqrt(sum((x-mean(x))^2))/length(x)};
der1 <- function(x) {y = x - lag(x);y[1] <- 0; return(y)};
der2 <- function(x) {y = der1(x) - lag(der1(x));y[1] <- 0; return(y)};
der1f <- function(x) {y = lead(x) - x; return(y)};
der2f <- function(x) {y = lead(der1f(x)) - der1f(x); return(y)};


Valid_Prediction <- vector(); wp_valid <- vector();
Test_Prediction <- read.csv("contribution_example.csv",sep = ";");


for (i in 1:6){
  traintest <- read.csv(paste0("wp",i,".csv"));
  traintest$datetime <- ymd_h(traintest$date) + hours(traintest$hors);
  traintest <- traintest[c(7,1:6)]
  traintest <- traintest %>% arrange(datetime, hors);
  
  a <- which(traintest$hors>12 & traintest$hors <25); tmpa <- traintest[a, ];
  b <- which(traintest$hors>24 & traintest$hors <37); tmpb <- traintest[b, ];
  c <- which(traintest$hors>36); tmpc <- traintest[c, ];
  traintest <- traintest[-c(a,b,c), ];
  colnames(tmpa) <- c("datetime", "date12", "hors12", "u12", "v12", "ws12", "wd12");
  colnames(tmpb) <- c("datetime", "date24", "hors24", "u24", "v24", "ws24", "wd24");
  colnames(tmpc) <- c("datetime", "date36", "hors36", "u36", "v36", "ws36", "wd36");
  
  
  traintest <- traintest %>% full_join(tmpa, by = "datetime") %>% 
    full_join(tmpb, by = "datetime") %>% full_join(tmpc, by = "datetime") 
  
  rm("tmpa","tmpb","tmpc","a","b","c")
  
  #traintest[inTrain_9, names(traintest) %in% c("u", "v", "ws", "wd")] <- NA;
  #traintest[inTrain_18, names(traintest) %in% c("u12", "v12", "ws12", "wd12")] <- NA;
  #traintest[inTrain_27, names(traintest) %in% c("u24", "v24", "ws24", "wd24")] <- NA;
  
  traintest <- traintest %>% mutate(unr = coalesce(u, u12, u24, u36));
  traintest <- traintest %>% mutate(vnr = coalesce(v, v12, v24, v36));
  traintest <- traintest %>% mutate(wsnr = coalesce(ws, ws12, ws24, ws36));
  traintest <- traintest %>% mutate(wdnr = coalesce(wd, wd12, wd24, wd36));
  
  traintest <- traintest %>% 
    mutate(u_av = rowMeans(cbind(u, u12, u24, u36), na.rm = TRUE));
  traintest <- traintest %>% 
    mutate(v_av = rowMeans(cbind(v, v12, v24, v36), na.rm = TRUE));
  traintest <- traintest %>% 
    mutate(ws_av = rowMeans(cbind(ws, ws12, ws24, ws36), na.rm = TRUE));
  traintest <- traintest %>% 
    mutate(wd_av = rowMeans(cbind(wd, wd12, wd24, wd36), na.rm = TRUE));
  
  traintest[26125:26160, c("date", "hors")] <- cbind(rep(c(2012062312, 2012062400, 2012062412), each = 12), seq(1,12));
  
  traintest <- subset(traintest, select = -c(u:wd36));
  
  traintest <- traintest %>% mutate(wsnr3 = wsnr^3, ws_av3 = ws_av^3);
  
  traintest <- traintest %>% left_join(traintest %>% group_by(date) %>% 
                                         mutate_at(vars(wsnr:wdnr, wsnr3), 
                                                   list(maf5 = rollapply), 
                                                   FUN = mean, width = 5, 
                                                   partial = TRUE, align = "left") %>% 
                                         ungroup %>%
                                         select(datetime, wsnr_maf5, wdnr_maf5, wsnr3_maf5), 
                                       by = "datetime");
  
  traintest <- traintest %>% left_join(traintest %>% group_by(date) %>% 
                                         mutate_at(vars(wsnr:wdnr, wsnr3), 
                                                   list(d1f = der1f, d2f = der2f)) %>% 
                                         ungroup %>%
                                         select(datetime, wsnr_d1f:wsnr3_d2f), 
                                       by = "datetime");
  
  #traintest <- traintest %>% left_join(
   # traintest %>% group_by(date) %>% summarise_at(vars(wsnr:wdnr, wsnr3), 
    #                                              list(av12 = mean, mx12 = max,
     #                                                  mn12 = min)), 
    #by = "date");
  
  traintest <- traintest %>%
    mutate_at(vars(wsnr:wdnr),list(ma4 = rollmeanr), k = 4, fill = 0, na.rm=TRUE);
  traintest <- traintest %>%
    mutate_at(vars(ws_av:wd_av),list(ma7 = rollmeanr), k = 7, fill = 0, na.rm=TRUE);
  traintest <- traintest %>%
    mutate_at(vars(wsnr3:ws_av3),list(ma10 = rollmeanr), k = 10, fill = 0, na.rm=TRUE);
  traintest <- traintest %>% 
    mutate_at(vars(wsnr:wdnr),list(ma13 = rollmeanr), k = 13, fill = 0, na.rm=TRUE);
  traintest <- traintest %>% mutate_at(vars(wsnr:wdnr, wsnr3),list(d1 = der1));
  traintest <- traintest %>% mutate_at(vars(wsnr:wdnr, wsnr3),list(d2 = der2));
  
  
  traintest <- traintest %>% 
    mutate_at(vars("datetime"), list(hour = hour, yday = yday, week = week, 
                                     wday = wday));
  
  
  assign(paste0("wp",i,"traintest"), traintest);
}


for (i in 1:6){
  
  traintest <- get(paste0("wp",i,"traintest"));
  
  if (i == 1){
    traintest <- traintest %>% left_join(wp4traintest, by = "datetime") %>% 
      left_join(wp6traintest, by = "datetime");
  }
  if (i == 4){
    traintest <- traintest %>% left_join(wp1traintest, by = "datetime");
  }
  
  if (i == 6){
    traintest <- traintest %>% left_join(wp1traintest, by = "datetime");
  }
  
  target <- trainds %>% select(datetime, wp = 1+i);
  traintest <- traintest %>% left_join(target, by = "datetime");
  
  
  inTrain <- 1:8760; inValid <- 8761:13140; st_inValid <- 13141:18720;  
  inTest <- which(is.na(traintest[,"wp"])); 
  testing <- traintest[inTest,]; trainvalid <- traintest[-inTest,]; 
  training <- trainvalid[inTrain,]; valid <- trainvalid[inValid,];
  st_valid <- trainvalid[st_inValid,]
  
  
  inTest <- 1:48; test <- testing[inTest,];
  
  
  for (j in 1:155){
    
    dtrain_cat <- catboost.load_pool(data.matrix(subset(training, select = -wp)), 
                                     label = training$wp^(1/4));
    dvalid_cat <- catboost.load_pool(data.matrix(subset(valid, select = -wp)), 
                                     label = valid$wp^(1/4));
    
    
    mdl_cat <- catboost.train(dtrain_cat, dvalid_cat, 
                              params = list(iterations = 750, learning_rate = 0.1, 
                                            depth = 4, logging_level = 'Silent', 
                                            l2_leaf_reg = 0.2, rsm = 1, 
                                            loss_function = 'MAE', od_type = 'Iter',
                                            od_wait = 75, subsample = 0.8));
    
    dtrain_lgbm <- lgb.Dataset(data =data.matrix(subset(training, select = -wp)), 
                               label = training$wp^(1/3));
    dvalid_lgbm <- lgb.Dataset(data =data.matrix(subset(valid, select = -wp)), 
                               label = valid$wp^(1/3));
    
    
    mdl_lgbm <- lightgbm(data = dtrain_lgbm, nrounds = 500, boosting_type = 'gbdt',
                         verbose = -1, learning_rate = 0.1, max_depth = 10, 
                         valids = list(valids = dvalid_lgbm),
                         obj = "regression_l1", early_stopping_rounds = 50);
    
    
    dtrain_xgb <- xgb.DMatrix(data =data.matrix(subset(training, select = -wp)), 
                              label = training$wp^(1/2));
    dvalid_xgb <- xgb.DMatrix(data =data.matrix(subset(valid, select = -wp)), 
                              label = valid$wp^(1/2));
    
    
    mdl_xgb <- xgb.train(data = dtrain_xgb, nrounds = 500, early_stopping_rounds = 50, 
                         watchlist = list(train = dtrain_xgb, eval = dvalid_xgb), verbose = 0,
                         params = list(max_depth = 4, eta = 0.11, gamma = 0.01,
                                       colsample_bytree = 1, min_child_weight = 0,
                                       subsample = 0.8, objective = "reg:pseudohubererror",
                                       eval_metric = "mae"));
    
    
    Valid_Pred_cat <- catboost.predict(mdl_cat, dvalid_cat)^4; 
    Valid_Pred_lgbm <- predict(mdl_lgbm, data.matrix(subset(valid, select = -wp)))^3; 
    Valid_Pred_xgb <- predict(mdl_xgb, dvalid_xgb)^2; 
    
    
    st_dtrain_xgb <- xgb.DMatrix(data = cbind(cat = Valid_Pred_cat, 
                                              lgbm = Valid_Pred_lgbm, 
                                              xgb = Valid_Pred_xgb),
                                 label = valid$wp^(1/2));
    
    dtest_xgb <- xgb.DMatrix(data =data.matrix(subset(test, select = -wp)));
    
    
    st_dvalid_cat <- catboost.load_pool(data.matrix(subset(st_valid, select = -wp)));
    #st_dvalid_lgbm <- data.matrix(subset(st_valid, select = -wp));
    st_dvalid_xgb <- xgb.DMatrix(data =data.matrix(subset(st_valid, select = -wp)));
    
    st_Valid_Pred_cat <- catboost.predict(mdl_cat, st_dvalid_cat)^4; 
    st_Valid_Pred_lgbm <- predict(mdl_lgbm, data.matrix(subset(st_valid, select = -wp)))^3; 
    st_Valid_Pred_xgb <- predict(mdl_xgb, st_dvalid_xgb)^2; 
    
    st_dvalid <- xgb.DMatrix(data = cbind(cat = st_Valid_Pred_cat, 
                                          lgbm = st_Valid_Pred_lgbm, 
                                          xgb = st_Valid_Pred_xgb),
                             label = st_valid$wp);
    
    
    st_mdl_xgb <- xgb.train(data = st_dtrain_xgb, nrounds = 150, verbose = 0,
                            early_stopping_rounds = 15, watchlist = 
                              list(train = st_dtrain_xgb, eval = st_dvalid),
                            params = list(max_depth = 2, eta = 0.11, gamma = 0.01,
                                          colsample_bytree = 1, min_child_weight = 0,
                                          subsample = 1, eval_metric = "mae",
                                          objective = "reg:pseudohubererror"));
    
    
    
    Valid_Pred <- predict(st_mdl_xgb, st_dvalid)^2;
    Valid_Pred[Valid_Pred < 0] <- 0; Valid_Pred[Valid_Pred > 1] <- 1;
    #print(cbind(ValidMAE = MAE(Valid_Pred,st_valid$wp), mdlMAE = st_mdl_xgb$best_score));
    #print(st_mdl_xgb$best_iteration)
    Valid_Prediction <- c(Valid_Prediction, Valid_Pred[1:36]); wp_valid <- c(wp_valid, st_valid$wp[1:36]);
    
    
    st_dtest_cat <- catboost.load_pool(data.matrix(subset(test, select = -wp)));
    st_dtest_lgbm <- data.matrix(subset(test, select = -wp));
    st_dtest_xgb <- xgb.DMatrix(data =data.matrix(subset(test, select = -wp)));
    
    Test_Pred_cat <- catboost.predict(mdl_cat, st_dtest_cat)^4; 
    Test_Pred_lgbm <- predict(mdl_lgbm, st_dtest_lgbm)^3;
    Test_Pred_xgb <- predict(mdl_xgb, st_dtest_xgb)^2;
    
    st_test <- xgb.DMatrix(data = cbind(cat = Test_Pred_cat, 
                                        lgbm = Test_Pred_lgbm, 
                                        xgb = Test_Pred_xgb));
    
    Test_Pred <- predict(st_mdl_xgb, st_test)^2;
    Test_Pred[Test_Pred < 0] <- 0;Test_Pred[Test_Pred > 1] <- 1;
    Test_Prediction[inTest, i+1] <- Test_Pred;
    
    if (j == 155)  {
      a <- length(wp_valid); b <- a+1-a/i; 
      print(MAE(Valid_Prediction[b:a], wp_valid[b:a]));
      break
    }
    
    training <- rbind(training, valid[1:36, ]);
    valid <- rbind(valid[-(1:36), ], st_valid[1:36, ]);
    st_inValid <- st_inValid[-(1:36)]; inTest <- inTest + 48; 
    st_valid <- trainvalid[st_inValid,]; test <- testing[inTest,];
  }
  
}

rm("a","b");
print(MAE(Valid_Prediction, wp_valid));
write.csv2(Test_Prediction, file = "Test_Results_10.csv", row.names = FALSE, quote = FALSE);
stopCluster(clus);
registerDoSEQ();
