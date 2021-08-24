library(lightgbm);library(parallel);library(doParallel);library(lubridate);
library(dplyr);library(zoo);library(caret);library(xgboost);library(catboost);
clus <- makeCluster(detectCores()-2);
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
  
  #traintest <- traintest %>% 
   # mutate(u_av = rowMeans(cbind(u, u12, u24, u36), na.rm = TRUE));
  #traintest <- traintest %>% 
   # mutate(v_av = rowMeans(cbind(v, v12, v24, v36), na.rm = TRUE));
  traintest <- traintest %>% 
    mutate(ws_av = rowMeans(cbind(ws, ws12, ws24, ws36), na.rm = TRUE));
  traintest <- traintest %>% 
    mutate(wd_av = rowMeans(cbind(wd, wd12, wd24, wd36), na.rm = TRUE));
  
  traintest <- subset(traintest, select = c(datetime:hors, ws36:wd36, ws_av:wd_av));
  
  traintest <- traintest %>%
    mutate_at(vars(ws_av:wd_av),list(ma5 = rollmeanr), k = 5, fill = 0, na.rm=TRUE);
  
  traintest <- traintest %>% 
    mutate_at(vars(ws_av:wd_av),list(ma13 = rollmeanr), k = 13, fill = 0, na.rm=TRUE);
  
  traintest <- traintest %>% mutate_at(vars(ws_av:wd_av),list(d1 = der1));
  traintest <- traintest %>% mutate(ws_av_d2 = der1(ws_av_d1));
  traintest <- traintest %>% mutate(wd_av_d2 = der1(wd_av_d1));
  
  traintest <- traintest %>% mutate(ws_av3 = ws_av^3);
  
  #traintest <- traintest %>% left_join(
  # traintest %>% group_by(date) %>% summarise_at(vars(ws36:wd36, ws_av:wd_av), list(mx12 = max, mn12 = min), na.rm=TRUE),
  #by = "date");
  
  traintest <- traintest %>% 
    mutate_at(vars("datetime"), list(hour = hour, yday = yday, week = week, wday = wday));
  
  target <- trainds %>% select(datetime, wp = 1+i);
  traintest <- traintest %>% left_join(target, by = "datetime");

  
  inTrain <- which(seq(26159,0) %% 84 > 47 & seq(1, 26160) < 13177);
  inValid <- which(seq(26159,0) %% 84 > 47 & seq(1, 26160) > 13176);
  inTest <- which(is.na(traintest[,"wp"])); 
  testing <- traintest[inTest,]; trainvalid <- traintest[-inTest,]; 
  training <- traintest[inTrain,]; valid <- traintest[inValid,]; 
  assign(paste0("wp",i,"trainvalid"),trainvalid);
  assign(paste0("wp",i,"train"),training); 
  assign(paste0("wp",i,"valid"),valid);
  assign(paste0("wp",i,"test"),testing);
  
  inTest <- 1:48; test <- testing[inTest,];
  
  
  for (j in 1:155){
    
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
    Valid_Prediction <- c(Valid_Prediction,Valid_Pred); wp_valid <- c(wp_valid,valid$wp);
    
    
    Test_Pred_cat <- catboost.predict(mdl_cat, dtest_cat)^4; 
    Test_Pred_lgbm <- predict(mdl_lgbm, dtest_lgbm)^3;
    Test_Pred_xgb <- predict(mdl_xgb, dtest_xgb)^2;
    
    Test_Pred <- (Test_Pred_cat + Test_Pred_lgbm + Test_Pred_xgb)/3;
    Test_Pred[Test_Pred < 0] <- 0;Test_Pred[Test_Pred > 1] <- 1;
    Test_Prediction[inTest, i+1] <- Test_Pred;
    
    if (j == 155)  {
      break
    }
    
    training <- rbind(training,valid[1:36, ]); 
    valid <- rbind(valid, training[1:36, ]);
    training <- training[-(1:36), ]; valid <- valid[-(1:36), ];
    inTest <- inTest + 48; test <- testing[inTest,];
  }
  print(MAE(Valid_Prediction,wp_valid));
  
  }

write.csv2(Test_Prediction, file = "Test_Results.csv", row.names = FALSE, quote = FALSE);
stopCluster(clus);
registerDoSEQ();
