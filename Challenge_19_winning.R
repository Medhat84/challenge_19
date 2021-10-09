#Calling required libraries and adjusting multi-core processing  
library(lightgbm);library(parallel);library(doParallel);library(lubridate);
library(dplyr);library(zoo);library(caret);library(xgboost);library(catboost);
clus <- makeCluster(detectCores()-0);
registerDoParallel(clus);

##################################################### Preprocessing ####################################################

#Setting working directory, reading wind power values, generating new datetime feature, rearranging columns and setting seed
setwd("~/R/R Directory/challenge_19");
trainds <- read.csv("train.csv"); testds <- read.csv("test.csv");
trainds$datetime <- ymd_h(trainds$date); trainds <- trainds[c(8,2:7)]
set.seed(145);

#Creating derivative functions for the past and the future
stdev <- function(x, ...) {x<- x[!is.na(x)];sqrt(sum((x-mean(x))^2))/length(x)};
der1 <- function(x) {y = x - lag(x);y[1] <- 0; return(y)};
der2 <- function(x) {y = der1(x) - lag(der1(x));y[1] <- 0; return(y)};
der1f <- function(x) {y = lead(x) - x; return(y)};
der2f <- function(x) {y = lead(der1f(x)) - der1f(x); return(y)};

#Initialization for some variables
Valid_Prediction <- vector(); wp_valid <- vector();
Test_Prediction <- read.csv("contribution_example.csv",sep = ";");

################################################# Features generation ####################################################

for (i in 1:6){
  #Reading training data, generating new datetime feature, rearranging columns and sorting rows according to time stamp 
  traintest <- read.csv(paste0("wp",i,".csv"));
  traintest$datetime <- ymd_h(traintest$date) + hours(traintest$hors);
  traintest <- traintest[c(7,1:6)]
  traintest <- traintest %>% arrange(datetime, hors);
  
  #As it's usually expected to have 4 rows with the same time stamp, they will be converted into columns to get 16 features instead of 4
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
  
  #As the forecasts from the first 12 hours will be more reliable than the second 12 hours and so on, new 4 feature will be created from
  #the nearest available forecast
  traintest <- traintest %>% mutate(unr = coalesce(u, u12, u24, u36));
  traintest <- traintest %>% mutate(vnr = coalesce(v, v12, v24, v36));
  traintest <- traintest %>% mutate(wsnr = coalesce(ws, ws12, ws24, ws36));
  traintest <- traintest %>% mutate(wdnr = coalesce(wd, wd12, wd24, wd36));
  
  #New 4 features are generated from the average of the 4 forecasts
  traintest <- traintest %>% 
    mutate(u_av = rowMeans(cbind(u, u12, u24, u36), na.rm = TRUE));
  traintest <- traintest %>% 
    mutate(v_av = rowMeans(cbind(v, v12, v24, v36), na.rm = TRUE));
  traintest <- traintest %>% 
    mutate(ws_av = rowMeans(cbind(ws, ws12, ws24, ws36), na.rm = TRUE));
  traintest <- traintest %>% 
    mutate(wd_av = rowMeans(cbind(wd, wd12, wd24, wd36), na.rm = TRUE));
  
  #Filling the last 36 rows of these 2 features as they have NA values
  traintest[26125:26160, c("date", "hors")] <- cbind(rep(c(2012062312, 2012062400, 2012062412), each = 12), seq(1,12));
  
  #Keeping only the 4 features with nearest forecasts and the 4 features with the forecasts average and 
  #excluding the other forecast features
  traintest <- subset(traintest, select = -c(u:wd36));
  
  #Generating features with the cube of wind speed
  traintest <- traintest %>% mutate(wsnr3 = wsnr^3, ws_av3 = ws_av^3);
  
  #Grouping the data by each 12 hours windows and generating forward moving average inside these 12 points windows,
  #this is not violation of the future usage rule as the data is updated at 12 hours intervals and at any given time, 
  #we always have the data until that limit 
  traintest <- traintest %>% left_join(traintest %>% group_by(date) %>% 
                                         mutate_at(vars(wsnr:wdnr, wsnr3), 
                                                   list(maf5 = rollapply), 
                                                   FUN = mean, width = 5, 
                                                   partial = TRUE, align = "left") %>% 
                                         ungroup %>%
                                         select(datetime, wsnr_maf5, wdnr_maf5, wsnr3_maf5), 
                                       by = "datetime");
  
  #Grouping the data by each 12 hours windows and generating forward 1st and 2nd derivatives. We will have NA at each 12 hour for
  #the 1st derivative and 2 NAs at each 12 hour for the 2nd derivative, but it's okay as it's improving the local validation score
  traintest <- traintest %>% left_join(traintest %>% group_by(date) %>% 
                                         mutate_at(vars(wsnr:wdnr, wsnr3), 
                                                   list(d1f = der1f, d2f = der2f)) %>% 
                                         ungroup %>%
                                         select(datetime, wsnr_d1f:wsnr3_d2f), 
                                       by = "datetime");
  
 
  #Generating different backward moving averages with different window size and different features
  traintest <- traintest %>%
    mutate_at(vars(wsnr:wdnr),list(ma4 = rollmeanr), k = 4, fill = 0, na.rm=TRUE);
  traintest <- traintest %>%
    mutate_at(vars(ws_av:wd_av),list(ma7 = rollmeanr), k = 7, fill = 0, na.rm=TRUE);
  traintest <- traintest %>%
    mutate_at(vars(wsnr3:ws_av3),list(ma10 = rollmeanr), k = 10, fill = 0, na.rm=TRUE);
  traintest <- traintest %>% 
    mutate_at(vars(wsnr:wdnr),list(ma13 = rollmeanr), k = 13, fill = 0, na.rm=TRUE);
  
  #Generating backward 1st and 2nd derivatives
  traintest <- traintest %>% mutate_at(vars(wsnr:wdnr, wsnr3),list(d1 = der1));
  traintest <- traintest %>% mutate_at(vars(wsnr:wdnr, wsnr3),list(d2 = der2));
  
  #Generating time related features; hour, day of year, week number, day of week and combining hour and day of year together
  traintest <- traintest %>% 
    mutate_at(vars("datetime"), list(hour = hour, yday = yday, week = week, 
                                     wday = wday));
  traintest <- traintest %>% mutate(yday_h = as.numeric(paste0(yday, hour)));
  
  #Creating new 6 data sets with the generated features
  assign(paste0("wp",i,"traintest"), traintest);
}

################################################# Modelling part ####################################################

for (i in 1:6){
  
  traintest <- get(paste0("wp",i,"traintest"));

  #As there is correlation found between farms number 1, 4 & 6; data sets of those farms will be combined together
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
  
  #Binding the target variable to the specified data set
  target <- trainds %>% select(datetime, wp = 1+i);
  traintest <- traintest %>% left_join(target, by = "datetime");
  
  #Splitting the overall data set into 3 data sets; training, validation & test
  #The initial training set is the data until the first hour of 1-Jan-2011, initial test set is the 48 hours after that 
  #until the first hour of 3-Jan-2011 and the initial validation set is combined of 8 periods of 36 hours after the 
  #initial test set (This is not violation of the future usage rule as this rule is removed in case of validation)
  inTrain <- 1:13176; inValid <- 13177:13464; 
  inTest <- which(is.na(traintest[,"wp"])); 
  testing <- traintest[inTest,]; trainvalid <- traintest[-inTest,]; 
  training <- trainvalid[inTrain,]; valid <- trainvalid[inValid,];
  inTest <- 1:48; test <- testing[inTest,];
  
  #Looping with the 155 test periods, modelling with 3 solvers; xgboost, catboost & light gbm then taking the 
  #average of the 3 solvers' predictions
  for (j in 1:155){
    
    #Generating catboost model
    dtrain_cat <- catboost.load_pool(data.matrix(subset(training, select = -wp)), 
                                     label = training$wp^(1/2));
    dvalid_cat <- catboost.load_pool(data.matrix(subset(valid, select = -wp)), 
                                     label = valid$wp^(1/2));
    dtest_cat <- catboost.load_pool(data.matrix(subset(test, select = -wp)));
    
    mdl_cat <- catboost.train(dtrain_cat, dvalid_cat, 
                              params = list(iterations = 750, learning_rate = 0.1, 
                                            depth = 4, logging_level = 'Silent', 
                                            l2_leaf_reg = 0.2, rsm = 1, 
                                            loss_function = 'MAE', od_type = 'Iter',
                                            od_wait = 75, subsample = 0.8));
    
    #Generating light gbm model
    dtrain_lgbm <- lgb.Dataset(data =data.matrix(subset(training, select = -wp)), 
                               label = training$wp^(1/3));
    dvalid_lgbm <- lgb.Dataset(data =data.matrix(subset(valid, select = -wp)), 
                               label = valid$wp^(1/3));
    dtest_lgbm <- data.matrix(subset(test, select = -wp));
    
    mdl_lgbm <- lightgbm(data = dtrain_lgbm, nrounds = 500, boosting_type = 'gbdt',
                         verbose = -1, learning_rate = 0.1, max_depth = 10, 
                         valids = list(valids = dvalid_lgbm),
                         obj = "regression_l1", early_stopping_rounds = 50);
    
    #Generating xgboost model
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
    
    #Generating validation predictions
    Valid_Pred_cat <- catboost.predict(mdl_cat, dvalid_cat)^2; 
    Valid_Pred_lgbm <- predict(mdl_lgbm, data.matrix(subset(valid, select = -wp)))^3; 
    Valid_Pred_xgb <- predict(mdl_xgb, dvalid_xgb)^2; 
    
    #Limiting validation predictions between zero & one
    Valid_Pred_cat[Valid_Pred_cat < 0] <- 0;Valid_Pred_cat[Valid_Pred_cat > 1] <- 1;
    Valid_Pred_lgbm[Valid_Pred_lgbm < 0] <- 0;Valid_Pred_lgbm[Valid_Pred_lgbm > 1] <- 1;
    Valid_Pred_xgb[Valid_Pred_xgb < 0] <- 0;Valid_Pred_xgb[Valid_Pred_xgb > 1] <- 1;
    #Calculating the average of the 3 predictions
    Valid_Pred <- (Valid_Pred_cat + Valid_Pred_lgbm + Valid_Pred_xgb)/3;
    Valid_Prediction <- c(Valid_Prediction,Valid_Pred[1:36]); wp_valid <- c(wp_valid,valid$wp[1:36]);
    
    #Generating test predictions
    Test_Pred_cat <- catboost.predict(mdl_cat, dtest_cat)^2; 
    Test_Pred_lgbm <- predict(mdl_lgbm, dtest_lgbm)^3;
    Test_Pred_xgb <- predict(mdl_xgb, dtest_xgb)^2;
    
    #Limiting test predictions between zero & one
    Test_Pred_cat[Test_Pred_cat < 0] <- 0;Test_Pred_cat[Test_Pred_cat > 1] <- 1;
    Test_Pred_lgbm[Test_Pred_lgbm < 0] <- 0;Test_Pred_lgbm[Test_Pred_lgbm > 1] <- 1;
    Test_Pred_xgb[Test_Pred_xgb < 0] <- 0;Test_Pred_xgb[Test_Pred_xgb > 1] <- 1;
    #Calculating the average of the 3 predictions 
    Test_Pred <- (Test_Pred_cat + Test_Pred_lgbm + Test_Pred_xgb)/3;
    Test_Prediction[inTest, i+1] <- Test_Pred;
    
    #Breaking the loop if we reached to the final test period and calculating the validation score for this wind farm
    if (j == 155)  {
      a <- length(wp_valid); b <- a+1-a/i; 
      print(MAE(Valid_Prediction[b:a], wp_valid[b:a]));
      break
    }
    
    #If we didn't reach the final 8 periods, updating the training set by taking the first 36 hours from validation set
    #and updating the validation set by increasing its index by 36
    if (j < 147) {
      training <- rbind(training,valid[1:36, ]); 
      inValid <- inValid+36; valid <- trainvalid[inValid,]
    }
    
    #If reached the final 8 periods, the validation set has to decrease by 36 points in each iteration
    if (j > 146 & j < 154) {
      training <- rbind(training,valid[1:36, ]); 
      inValid <- inValid[-(1:36)]; valid <- trainvalid[inValid,]
    }
    
    #Test data set index is increased by 48 in each iteration
    inTest <- inTest + 48; test <- testing[inTest,];
  }
  
}

#Calculating the overall validation score and exporting the test predictions for submission 
rm("a","b");
print(MAE(Valid_Prediction, wp_valid));
write.csv2(Test_Prediction, file = "Test_Results_9.csv", row.names = FALSE, quote = FALSE);
stopCluster(clus);
registerDoSEQ();
