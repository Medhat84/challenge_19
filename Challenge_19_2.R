library(parallel);library(doParallel);library(lubridate);library(dplyr);
library(zoo);library(caret);
clus <- makeCluster(detectCores()-2);
registerDoParallel(clus);
setwd("~/R/R Directory/challenge_19");
trainds <- read.csv("train.csv"); testds <- read.csv("test.csv");
trainds$date <- ymd_h(trainds$date); testds$date <- ymd_h(testds$date);
set.seed(145);

for (i in 1:6){
  traintest <- read.csv(paste0("wp",i,".csv"));
  traintest$date <- ymd_h(traintest$date) + hours(traintest$hors);
  traintest <- traintest %>% arrange(date, hors);
  traintest <- traintest %>% group_by(date) %>% summarise(uav = mean(u, na.rm=TRUE),
                                              vav = mean(v, na.rm=TRUE),
                                              wsav = mean(ws, na.rm=TRUE),
                                              wdav = mean(wd, na.rm=TRUE));
  
  target <- trainds %>% select(date,wp=1+i);
  traintest <- traintest %>% left_join(target, by = "date");
  traintest$hour <- hour(traintest$date);
  traintest$wday <- wday(traintest$date);
  traintest <- traintest %>% mutate(wsav_ma5 = rollmeanr(wsav, k = 5, fill = 0));
  traintest <- traintest %>% mutate(wsav_ma4 = rollmeanr(wsav, k = 4, fill = 0));
  traintest <- traintest %>% mutate(wsav_ma3 = rollmeanr(wsav, k = 3, fill = 0));
  traintest <- traintest %>% mutate(wp_ma = rollmeanr(wp, k = 49, fill = 0, na.rm=TRUE));
  traintest <- traintest %>% mutate(wsav_dt1 = wsav - lag(wsav));traintest$wsav_dt1[1] <- 0; 
  traintest <- traintest %>% mutate(wsav_dt2 = wsav_dt1 - lag(wsav_dt1));traintest$wsav_dt2[1] <- 0; 
  traintest <- traintest %>% mutate(wsav_msd = rollapplyr(wsav, 4, sd, fill = 0));
  traintest <- traintest %>% mutate(wp_ma49_ma8 = rollmeanr(wp_ma, k = 8, fill = 0, na.rm=TRUE));
  

  
  
  inTest <- which(is.na(traintest[,"wp"])); inValid <- inTest-length(testds$date);
  trainvalid <- traintest[-inTest,]; training <- trainvalid[-inValid,]; 
  valid <- trainvalid[inValid,]; test <- traintest[inTest,];
  assign(paste0("wp",i,"trainvalid"),trainvalid);
  assign(paste0("wp",i,"train"),training); 
  assign(paste0("wp",i,"valid"),valid);
  assign(paste0("wp",i,"test"),test);
  
  #mdl <- train(wp ~ ., method = "gbm", data = training, distribution = "laplace",
  #             metric="MAE",verbose=FALSE,tuneGrid=expand.grid(n.trees=150,
  #                                                             interaction.depth=4,
  #                                                             shrinkage=0.1,
  #                                                             n.minobsinnode=6));
  
  #Valid_Pred <- predict(mdl, valid); 
  #Valid_Pred[Valid_Pred < 0] <- 0;Valid_Pred[Valid_Pred > 1] <- 1;
  #print(cbind(ValidMAE = MAE(Valid_Pred,valid$wp), mdl$results[7]));
  #assign(paste0("Valid_Pred",i),Valid_Pred);
  #Test_Pred <- predict(mdl, test); 
  #Test_Pred[Test_Pred < 0] <- 0;Test_Pred[Test_Pred > 1] <- 1;
  #assign(paste0("Test_Pred",i),Test_Pred); 
  }

#Valid_Pred <- c(Valid_Pred1,Valid_Pred2,Valid_Pred3,Valid_Pred4,Valid_Pred5,Valid_Pred6);
#wp_valid <- c(wp1valid$wp,wp2valid$wp,wp3valid$wp,wp4valid$wp,wp5valid$wp,wp6valid$wp);
#print(MAE(Valid_Pred,wp_valid));
#Test_Pred <- cbind(Test_Pred1,Test_Pred2,Test_Pred3,Test_Pred4,Test_Pred5,Test_Pred6);
#samp_sol <- read.csv("contribution_example.csv",sep = ";");
#Result <- cbind(samp_sol$date,Test_Pred);
#colnames(Result) <- c("date","wp1","wp2","wp3","wp4","wp5","wp6");
#write.csv2(Result, file = "Test_Results.csv", row.names = FALSE, quote = FALSE);
stopCluster(clus);
registerDoSEQ();
