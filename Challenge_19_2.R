library(parallel);library(doParallel);library(lubridate);library(dplyr);
library(zoo);library(caret);
clus <- makeCluster(detectCores()-2);
registerDoParallel(clus);
setwd("~/R/R Directory/challenge_19");
trainds <- read.csv("train.csv"); testds <- read.csv("test.csv");
trainds$date <- ymd_h(trainds$date); testds$date <- ymd_h(testds$date);
set.seed(145);

inValid_starts <- -11:36; inValid <- inValid_starts;
inValid_starts_12 <- 1:36; inValid_12 <- inValid_starts_12;
inValid_starts_24 <- 13:36; inValid_24 <- inValid_starts_24;
inValid_starts_36 <- 25:36; inValid_36 <- inValid_starts_36;

for (i in 1:156){inValid <- c(inValid,inValid_starts+84*i);
                 inValid_12 <- c(inValid_12,inValid_starts_12+84*i);
                 inValid_24 <- c(inValid_24,inValid_starts_24+84*i);
                 inValid_36 <- c(inValid_36,inValid_starts_36+84*i)};

inValid <- inValid[inValid>0]; inValid_12 <- inValid_12[inValid_12>36];
inValid_24 <- inValid_24[inValid_24>36]; inValid_36 <- inValid_36[inValid_36>36];
inValid_dates_12 <- trainds$date[inValid_12+1]; 
inValid_dates_24 <- trainds$date[inValid_24+1];
inValid_dates_36 <- trainds$date[inValid_36+1];


stdev <- function(x, ...) {x<- x[!is.na(x)];sqrt(sum((x-mean(x))^2))/length(x)};
der1 <- function(x) {y = x - lag(x);y[1] <- 0; return(y)};

for (i in 1:6){
  traintest <- read.csv(paste0("wp",i,".csv"));
  traintest$date <- ymd_h(traintest$date) + hours(traintest$hors);
  traintest <- traintest %>% arrange(date, hors);
  traintest[(traintest$date %in% inValid_dates_12) & traintest$hors<13, 3:6] <- NA;
  traintest[(traintest$date %in% inValid_dates_24) & traintest$hors<25 & traintest$hors>12, 3:6] <- NA;
  traintest[(traintest$date %in% inValid_dates_36) & traintest$hors<37 & traintest$hors>24, 3:6] <- NA;
  traintest <- traintest %>% group_by(date) %>% summarise(ws_av = mean(ws, na.rm=TRUE),
                                                          wd_av = mean(wd, na.rm=TRUE),
                                                          u_av = mean(u, na.rm=TRUE),
                                                          v_av = mean(v, na.rm=TRUE));
  
  target <- trainds %>% select(date,wp=1+i);
  traintest <- traintest %>% left_join(target, by = "date");
  traintest$hour <- hour(traintest$date);
  traintest$yday <- yday(traintest$date);
  traintest <- traintest %>% mutate(ws_av_ma4 = rollmeanr(ws_av, k = 4, fill = 0));
  traintest <- traintest %>% mutate(ws_av_ma13 = rollmeanr(ws_av, k = 13, fill = 0));
  #traintest <- traintest %>% mutate(wp_ma61 = rollmeanr(wp, k = 61, fill = 0, na.rm=TRUE));
  #traintest <- traintest %>% mutate(wp_ma61_ma8 = rollmeanr(wp_ma61, k = 8, fill = 0, na.rm=TRUE));
  traintest <- traintest %>% mutate(ws_av_d1 = der1(ws_av));
  traintest <- traintest %>% mutate(ws_av_d2 = der1(ws_av_d1));
 

  
  
  inTest <- which(is.na(traintest[,"wp"])); 
  trainvalid <- traintest[-inTest,]; training <- trainvalid[-inValid,]; 
  valid <- trainvalid[inValid,]; test <- traintest[inTest,];
  assign(paste0("wp",i,"trainvalid"),trainvalid);
  assign(paste0("wp",i,"train"),training); 
  assign(paste0("wp",i,"valid"),valid);
  assign(paste0("wp",i,"test"),test);
  
  mdl <- train(wp ~ ., method = "gbm", data = training, distribution = "laplace",
               metric="MAE",verbose=FALSE,tuneGrid=expand.grid(n.trees=150,
                                                               interaction.depth=4,
                                                               shrinkage=0.1,
                                                               n.minobsinnode=6));
  
  Valid_Pred <- predict(mdl, valid); 
  Valid_Pred[Valid_Pred < 0] <- 0;Valid_Pred[Valid_Pred > 1] <- 1;
  print(cbind(ValidMAE = MAE(Valid_Pred,valid$wp), mdlMAE = mdl$results$MAE));
  assign(paste0("Valid_Pred",i),Valid_Pred);
  Test_Pred <- predict(mdl, test); 
  Test_Pred[Test_Pred < 0] <- 0;Test_Pred[Test_Pred > 1] <- 1;
  assign(paste0("Test_Pred",i),Test_Pred); 
  }

Valid_Pred <- c(Valid_Pred1,Valid_Pred2,Valid_Pred3,Valid_Pred4,Valid_Pred5,Valid_Pred6);
wp_valid <- c(wp1valid$wp,wp2valid$wp,wp3valid$wp,wp4valid$wp,wp5valid$wp,wp6valid$wp);
print(MAE(Valid_Pred,wp_valid));
Test_Pred <- cbind(Test_Pred1,Test_Pred2,Test_Pred3,Test_Pred4,Test_Pred5,Test_Pred6);
samp_sol <- read.csv("contribution_example.csv",sep = ";");
Result <- cbind(samp_sol$date,Test_Pred);
colnames(Result) <- c("date","wp1","wp2","wp3","wp4","wp5","wp6");
write.csv2(Result, file = "Test_Results.csv", row.names = FALSE, quote = FALSE);
stopCluster(clus);
registerDoSEQ();
