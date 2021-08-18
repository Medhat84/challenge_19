library(lightgbm);library(parallel);library(doParallel);library(lubridate);
library(dplyr);library(zoo);library(caret);library(xgboost);library(catboost);
clus <- makeCluster(detectCores()-2);
registerDoParallel(clus);

setwd("~/R/R Directory/challenge_19");
trainds <- read.csv("train.csv"); testds <- read.csv("test.csv");
trainds$date <- ymd_h(trainds$date); testds$date <- ymd_h(testds$date);
set.seed(145);

inValid <- which(seq(13139,0) %% 168 < 48);
inValid_12 <- setdiff(which(seq(13139,0) %% 168 < 36), 1:36);
inValid_24 <- setdiff(which(seq(13139,0) %% 168 < 24), 1:36);
inValid_36 <- setdiff(which(seq(13139,0) %% 168 < 12), 1:36);

st_inValid <- which(seq(13139,0) %% 168 < 132 & seq(13139,0) %% 168 > 83);
#inValid_12 <- which(seq(13139,0) %% 168 < 120 & seq(13139,0) %% 168 > 83);
#inValid_24 <- which(seq(13139,0) %% 168 < 108 & seq(13139,0) %% 168 > 83);
#inValid_36 <- which(seq(13139,0) %% 168 < 96 & seq(13139,0) %% 168 > 83);

stdev <- function(x, ...) {x<- x[!is.na(x)];sqrt(sum((x-mean(x))^2))/length(x)};
der1 <- function(x) {y = x - lag(x);y[1] <- 0; return(y)};

Valid_Prediction <- vector(); wp_valid <- vector();
Test_Prediction <- read.csv("contribution_example.csv",sep = ";");

for (i in 1:6){
  traintest <- read.csv(paste0("wp",i,".csv"));
  traintest$date <- ymd_h(traintest$date) + hours(traintest$hors);
  traintest <- traintest %>% arrange(date, hors);
  traintest[(traintest$date %in% trainds$date[inValid_12+1]) & traintest$hors<13, 3:6] <- NA;
  traintest[(traintest$date %in% trainds$date[inValid_24+1]) & traintest$hors<25 & traintest$hors>12, 3:6] <- NA;
  traintest[(traintest$date %in% trainds$date[inValid_36+1]) & traintest$hors<37 & traintest$hors>24, 3:6] <- NA;
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
  
  traintest <- traintest %>% 
    mutate_at(vars("date"), list(yday = yday, hour = hour));
  
  
  inTest <- which(is.na(traintest[,"wp"]));
  trainvalid <- traintest[-inTest,]; training <- trainvalid[-inValid,]; 
  valid <- trainvalid[inValid,]; test <- traintest[inTest,];
  st_valid <- trainvalid[st_inValid,]
  assign(paste0("wp",i,"trainvalid"),trainvalid);
  assign(paste0("wp",i,"train"),training); 
  assign(paste0("wp",i,"valid"),valid);
  assign(paste0("wp",i,"test"),test);
  assign(paste0("wp",i,"valid"),st_valid);
  
  dtrain_cat <- catboost.load_pool(data.matrix(subset(training, select = -wp)), 
                                   label = training$wp^(1/4));
  dvalid_cat <- catboost.load_pool(data.matrix(subset(valid, select = -wp)), 
                                   label = valid$wp^(1/4));
  
  
  mdl_cat <- catboost.train(dtrain_cat, dvalid_cat, 
                            params = list(iterations = 300, learning_rate = 0.1, 
                                          depth = 6, logging_level = 'Silent', 
                                          l2_leaf_reg = 0.1, rsm = 1, 
                                          loss_function = 'MAE', 
                                          od_type = 'Iter', od_wait = 30));
  
  dtrain_lgbm <- lgb.Dataset(data =data.matrix(subset(training, select = -wp)), 
                             label = training$wp^(1/3));
  dvalid_lgbm <- lgb.Dataset(data =data.matrix(subset(valid, select = -wp)), 
                             label = valid$wp^(1/3));
  
  
  mdl_lgbm <- lightgbm(data = dtrain_lgbm, nrounds = 150, boosting_type = 'gbdt',
                       verbose = -1, learning_rate = 0.1, max_depth = 8, 
                       valids = list(valids = dvalid_lgbm),
                       obj = "regression_l1", early_stopping_rounds = 15);
  
  dtrain_xgb <- xgb.DMatrix(data =data.matrix(subset(training, select = -wp)), 
                            label = training$wp^(1/2));
  dvalid_xgb <- xgb.DMatrix(data =data.matrix(subset(valid, select = -wp)), 
                            label = valid$wp^(1/2));
  
  
  mdl_xgb <- xgb.train(data = dtrain_xgb, nrounds = 200, early_stopping_rounds = 20, 
                       watchlist = list(train = dtrain_xgb, eval = dvalid_xgb), verbose = 0,
                       params = list(max_depth = 4, eta = 0.11, gamma = 0.01,
                                     colsample_bytree = 1, min_child_weight = 0,
                                     subsample = 0.8, eval_metric = "mae",
                                     objective = "reg:pseudohubererror"));
  
  
  Valid_Pred_cat <- catboost.predict(mdl_cat, dvalid_cat)^4; 
  Valid_Pred_lgbm <- predict(mdl_lgbm, data.matrix(subset(valid, select = -wp)))^3; 
  Valid_Pred_xgb <- predict(mdl_xgb, dvalid_xgb)^2; 
  
 # st_train <- data.frame(cat = Valid_Pred_cat, lgbm = Valid_Pred_lgbm, 
  #                       xgb = Valid_Pred_xgb, wp = valid$wp);
  
  st_dtrain_xgb <- xgb.DMatrix(data = cbind(cat = Valid_Pred_cat, 
                                            lgbm = Valid_Pred_lgbm, 
                                            xgb = Valid_Pred_xgb),
                               label = valid$wp^(1/2));
  
  dtest_xgb <- xgb.DMatrix(data =data.matrix(subset(test, select = -wp)));
  
  
  st_mdl_xgb <- xgb.train(data = st_dtrain_xgb, nrounds = 85, verbose = 0,
                          params = list(max_depth = 4, eta = 0.11, gamma = 0.01,
                                        colsample_bytree = 1, min_child_weight = 0,
                                        subsample = 0.9, eval_metric = "mae",
                                        objective = "reg:pseudohubererror"));
  
  
  st_dvalid_cat <- catboost.load_pool(data.matrix(subset(st_valid, select = -wp)));
  st_dvalid_lgbm <- data.matrix(subset(st_valid, select = -wp));
  st_dvalid_xgb <- xgb.DMatrix(data =data.matrix(subset(st_valid, select = -wp)));
  
  st_Valid_Pred_cat <- catboost.predict(mdl_cat, st_dvalid_cat)^4; 
  st_Valid_Pred_lgbm <- predict(mdl_lgbm, data.matrix(subset(st_valid, select = -wp)))^3; 
  st_Valid_Pred_xgb <- predict(mdl_xgb, st_dvalid_xgb)^2; 
  
  st_dvalid <- xgb.DMatrix(data = cbind(cat = st_Valid_Pred_cat, 
                                        lgbm = st_Valid_Pred_lgbm, 
                                        xgb = st_Valid_Pred_xgb));
  
  Valid_Pred <- predict(st_mdl_xgb, st_dvalid)^2;
  Valid_Pred[Valid_Pred < 0] <- 0;Valid_Pred[Valid_Pred > 1] <- 1;
  print(MAE(Valid_Pred, st_valid$wp));
  Valid_Prediction <- c(Valid_Prediction,Valid_Pred); wp_valid <- c(wp_valid,st_valid$wp);

  
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
  Test_Prediction[,i+1] <- Test_Pred;
  }

print(MAE(Valid_Prediction, wp_valid));
write.csv2(Test_Prediction, file = "Test_Results.csv", row.names = FALSE, quote = FALSE);
stopCluster(clus);
registerDoSEQ();
