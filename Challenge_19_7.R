library(lightgbm);library(parallel);library(doParallel);library(lubridate);
library(dplyr);library(zoo);library(caret);library(xgboost);library(catboost);
clus <- makeCluster(detectCores()-2);
registerDoParallel(clus);

setwd("~/R/R Directory/challenge_19");
trainds <- read.csv("train.csv"); testds <- read.csv("test.csv");
trainds$date <- ymd_h(trainds$date); testds$date <- ymd_h(testds$date);
set.seed(145);

#inValid <- 6552+which(seq(6623,0) %% 96 < 48);
#inValid_12 <- 6552+which(seq(6623,0) %% 96 < 36);
#inValid_24 <- 6552+which(seq(6623,0) %% 96 < 24);
#inValid_36 <- 6552+which(seq(6623,0) %% 96 < 12);

#inValid_12 <- 13141 + which(seq(18720,13142) %% 36 < 27);
#inValid_24 <- 13141 + which(seq(18720,13142) %% 36 < 18);
#inValid_36 <- 13141 + which(seq(18720,13142) %% 36 < 9);

stdev <- function(x, ...) {x<- x[!is.na(x)];sqrt(sum((x-mean(x))^2))/length(x)};
der1 <- function(x) {y = x - lag(x);y[1] <- 0; return(y)};

Valid_Prediction <- vector(); wp_valid <- vector();
Test_Prediction <- read.csv("contribution_example.csv",sep = ";");


for (i in 1:1){
  traintest <- read.csv(paste0("wp",i,".csv"));
  traintest$date <- ymd_h(traintest$date) + hours(traintest$hors);
  traintest <- traintest %>% arrange(date, hors);
  
  #traintest[(traintest$date %in% trainds$date[inValid_12+1]) & traintest$hors<13, 3:6] <- NA;
  #traintest[(traintest$date %in% trainds$date[inValid_24+1]) & traintest$hors<25 & traintest$hors>12, 3:6] <- NA;
  #traintest[(traintest$date %in% trainds$date[inValid_36+1]) & traintest$hors<37 & traintest$hors>24, 3:6] <- NA;
  
  traintest <- traintest %>% group_by(date) %>%  
    summarise_all(list(av = mean), na.rm=TRUE);
                                              
  target <- trainds %>% select(date,wp=1+i);
  traintest <- traintest %>% left_join(target, by = "date");
  
  traintest <- traintest %>% 
    mutate_at(vars(ws_av:wd_av),list(ma4 = rollmeanr),k = 4, fill = 0, na.rm=TRUE);
  
  traintest <- traintest %>% 
    mutate_at(vars(ws_av:wd_av),list(ma13 = rollmeanr),k = 13, fill = 0, na.rm=TRUE);

  traintest <- traintest %>% mutate_at(vars(ws_av:wd_av),list(d1 = der1));
  traintest <- traintest %>% mutate(ws_av_d2 = der1(ws_av_d1));
  traintest <- traintest %>% mutate(wd_av_d2 = der1(wd_av_d1));
  
  traintest <- traintest %>% mutate(ws_av3 = ws_av^3);
 
  
  traintest <- traintest %>% 
    mutate_at(vars("date"), list(yday = yday, hour = hour, week = week, wday = wday));
  
  traintest <- traintest %>% mutate(yday_hour = as.numeric(paste0(yday,hour)));

  
  inValid <- 13141:13176; inTrain <- 1:13140;
  inTest <- which(is.na(traintest[,"wp"])); 
  testing <- traintest[inTest,]; trainvalid <- traintest[-inTest,]; 
  training <- trainvalid[inTrain,]; valid <- trainvalid[inValid,]; 
  assign(paste0("wp",i,"trainvalid"),trainvalid);
  assign(paste0("wp",i,"train"),training); 
  assign(paste0("wp",i,"valid"),valid);
  assign(paste0("wp",i,"test"),testing);
  
  inTest <- 1:48; test <- testing[inTest,];
  
  
  dtrain_cat <- catboost.load_pool(data.matrix(subset(training, select = -wp)), 
                                   label = training$wp^(1/4));
  dvalid_cat <- catboost.load_pool(data.matrix(subset(valid, select = -wp)), 
                                   label = valid$wp^(1/4));
  dtest_cat <- catboost.load_pool(data.matrix(subset(test, select = -wp)));
  
  mdl_cat <- catboost.train(dtrain_cat, dvalid_cat, 
                            params = list(iterations = 1000, learning_rate = 0.1, 
                                          depth = 4, logging_level = 'Silent', 
                                          l2_leaf_reg = 0.2, rsm = 1, 
                                          loss_function = 'MAE', od_type = 'Iter',
                                          od_wait = 100, subsample = 0.8));
  
  dtrain_lgbm <- lgb.Dataset(data =data.matrix(subset(training, select = -wp)), 
                             label = training$wp^(1/3));
  dvalid_lgbm <- lgb.Dataset(data =data.matrix(subset(valid, select = -wp)), 
                             label = valid$wp^(1/3));
  dtest_lgbm <- data.matrix(subset(test, select = -wp));
  
  mdl_lgbm <- lightgbm(data = dtrain_lgbm, nrounds = 150, boosting_type = 'gbdt',
                       verbose = -1, learning_rate = 0.1, max_depth = 10, 
                       valids = list(valids = dvalid_lgbm),
                       obj = "regression_l1", early_stopping_rounds = 15);
  
  dtrain_xgb <- xgb.DMatrix(data =data.matrix(subset(training, select = -wp)), 
                            label = training$wp^(1/2));
  dvalid_xgb <- xgb.DMatrix(data =data.matrix(subset(valid, select = -wp)), 
                            label = valid$wp^(1/2));
  dtest_xgb <- xgb.DMatrix(data =data.matrix(subset(test, select = -wp)));
  
  mdl_xgb <- xgb.train(data = dtrain_xgb, nrounds = 500, early_stopping_rounds = 50, 
                       watchlist = list(train = dtrain_xgb, eval = dvalid_xgb), verbose = 0,
                       params = list(max_depth = 4, eta = 0.11, gamma = 0.01,
                                     colsample_bytree = 1, min_child_weight = 0,
                                     subsample = 0.8, objective = "reg:pseudohubererror",
                                     eval_metric = "mae"));
  
  
  Valid_Pred_cat <- catboost.predict(mdl_cat, dvalid_cat)^4; 
  Valid_Pred_lgbm <- predict(mdl_lgbm, data.matrix(subset(valid, select = -wp)))^3; 
  Valid_Pred_xgb <- predict(mdl_xgb, dvalid_xgb)^2; 
  
  
  Valid_Pred <- (Valid_Pred_cat + Valid_Pred_lgbm + Valid_Pred_xgb)/3;
  Valid_Pred[Valid_Pred < 0] <- 0;Valid_Pred[Valid_Pred > 1] <- 1;
  #print(MAE(Valid_Pred, valid$wp));
  Valid_Prediction <- c(Valid_Prediction,Valid_Pred); wp_valid <- c(wp_valid,valid$wp);

  
  Test_Pred_cat <- catboost.predict(mdl_cat, dtest_cat)^4; 
  Test_Pred_lgbm <- predict(mdl_lgbm, dtest_lgbm)^3;
  Test_Pred_xgb <- predict(mdl_xgb, dtest_xgb)^2;
  
  Test_Pred <- (Test_Pred_cat + Test_Pred_lgbm + Test_Pred_xgb)/3;
  Test_Pred[Test_Pred < 0] <- 0;Test_Pred[Test_Pred > 1] <- 1;
  Test_Prediction[,i+1] <- Test_Pred;
  
  for (j in 1:154){
    training <- rbind(training,valid); 
    inValid <- inValid + 36; inTest <- inTest + 48; 
    valid <- trainvalid[inValid,]; test <- testing[inTest,];
    
    dtrain_cat <- catboost.load_pool(data.matrix(subset(training, select = -wp)), 
                                     label = training$wp^(1/4));
    dvalid_cat <- catboost.load_pool(data.matrix(subset(valid, select = -wp)), 
                                     label = valid$wp^(1/4));
    dtest_cat <- catboost.load_pool(data.matrix(subset(test, select = -wp)));
    
    mdl_cat <- catboost.train(dtrain_cat, dvalid_cat, 
                              params = list(iterations = 1000, learning_rate = 0.1, 
                                            depth = 4, logging_level = 'Silent', 
                                            l2_leaf_reg = 0.2, rsm = 1, 
                                            loss_function = 'MAE', od_type = 'Iter',
                                            od_wait = 100, subsample = 0.8));
    
    dtrain_lgbm <- lgb.Dataset(data =data.matrix(subset(training, select = -wp)), 
                               label = training$wp^(1/3));
    dvalid_lgbm <- lgb.Dataset(data =data.matrix(subset(valid, select = -wp)), 
                               label = valid$wp^(1/3));
    dtest_lgbm <- data.matrix(subset(test, select = -wp));
    
    mdl_lgbm <- lightgbm(data = dtrain_lgbm, nrounds = 150, boosting_type = 'gbdt',
                         verbose = -1, learning_rate = 0.1, max_depth = 10, 
                         valids = list(valids = dvalid_lgbm),
                         obj = "regression_l1", early_stopping_rounds = 15);
    
    dtrain_xgb <- xgb.DMatrix(data =data.matrix(subset(training, select = -wp)), 
                              label = training$wp^(1/2));
    dvalid_xgb <- xgb.DMatrix(data =data.matrix(subset(valid, select = -wp)), 
                              label = valid$wp^(1/2));
    dtest_xgb <- xgb.DMatrix(data =data.matrix(subset(test, select = -wp)));
    
    mdl_xgb <- xgb.train(data = dtrain_xgb, nrounds = 500, early_stopping_rounds = 50, 
                         watchlist = list(train = dtrain_xgb, eval = dvalid_xgb), 
                         verbose = 0, 
                         params = list(max_depth = 4, eta = 0.11, gamma = 0.01,
                                       colsample_bytree = 1, min_child_weight = 0,
                                       subsample = 0.8, objective = "reg:pseudohubererror",
                                       eval_metric = "mae"));
    
    
    Valid_Pred_cat <- catboost.predict(mdl_cat, dvalid_cat)^4; 
    Valid_Pred_lgbm <- predict(mdl_lgbm, data.matrix(subset(valid, select = -wp)))^3; 
    Valid_Pred_xgb <- predict(mdl_xgb, dvalid_xgb)^2; 
    
    
    Valid_Pred <- (Valid_Pred_cat + Valid_Pred_lgbm + Valid_Pred_xgb)/3;
    Valid_Pred[Valid_Pred < 0] <- 0;Valid_Pred[Valid_Pred > 1] <- 1;
    #print(MAE(Valid_Pred, valid$wp));
    Valid_Prediction <- c(Valid_Prediction,Valid_Pred); wp_valid <- c(wp_valid,valid$wp);
    
    
    Test_Pred_cat <- catboost.predict(mdl_cat, dtest_cat)^4; 
    Test_Pred_lgbm <- predict(mdl_lgbm, dtest_lgbm)^3;
    Test_Pred_xgb <- predict(mdl_xgb, dtest_xgb)^2;
    
    Test_Pred <- (Test_Pred_cat + Test_Pred_lgbm + Test_Pred_xgb)/3;
    Test_Pred[Test_Pred < 0] <- 0;Test_Pred[Test_Pred > 1] <- 1;
    Test_Prediction[,i+1] <- Test_Pred;
  }
  print(MAE(Valid_Prediction,wp_valid));
  
  }

#print(MAE(Valid_Prediction,wp_valid));
write.csv2(Test_Prediction, file = "Test_Results.csv", row.names = FALSE, quote = FALSE);
stopCluster(clus);
registerDoSEQ();
