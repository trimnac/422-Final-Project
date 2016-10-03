library(e1071)
library(car)
library(randomForest)
library(tree)
library(gbm)
library(doMC)
library(foreach)

charity <- read.csv('Dropbox/NU/PREDICT 422/Final/charity.csv')
charity$donr = as.factor(charity$donr)

data.train = charity[charity$part=='train',!colnames(charity) %in% c('ID','part')]
reg.data.train = data.train[data.train$damt > 0,!colnames(data.train) %in% c('donr')]
x.train = data.train[,1:20]
x.train.mean <- apply(x.train, 2, mean)
x.train.sd <- apply(x.train, 2, sd)
x.train.std <- t((t(x.train)-x.train.mean)/x.train.sd)
#apply(x.train.std, 2, mean) # check zero mean
#apply(x.train.std, 2, sd) # check unit sd

data.val = charity[charity$part=='valid',!colnames(charity) %in% c('ID','part')]
reg.data.val = data.val[data.val$damt > 0,!colnames(data.val) %in% c('donr')]
x.val = data.val[,1:20]

c.valid = as.integer(data.val[,21]) - 1
c.train = as.integer(data.train[,21]) - 1

y.train <- data.train[c.train==1,22]
y.valid <- data.val[c.valid==1,22]


data.train.std.c <- data.frame(x.train.std, donr=c.train) # to classify donr
data.train.std.y <- data.frame(x.train.std[c.train==1,], damt=y.train) # to predict damt when donr=1

x.valid.std <- t((t(x.val)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.valid.std.c <- data.frame(x.valid.std, donr=c.valid) # to classify donr
data.valid.std.y <- data.frame(x.valid.std[c.valid==1,], damt=y.valid) # to predict damt when donr=1

#Logistic Regression GAM
variables = c('chld', 'hinc', 'wrat', 'avhv', 'incm', 'inca', 'plow',
              'npro', 'tgif', 'lgif', 'rgif', 'tdon', 'tlag', 'agif')

best_poly = matrix(nrow=4, ncol=14)
for (i in 1:14){
  for (j in 1:4){
    gam.fit = glm(as.formula(paste('donr~poly(', variables[i], ', ', j, ')')), 
                 data=data.train.std.c,
                 family=binomial("logit"))
    
    gam.pred = predict(gam.fit, data.valid.std.c, type="response")
    profit.gam <- cumsum(14.5*c.valid[order(gam.pred, decreasing=T)]-2)
    
    best_poly[j,i] = max(profit.gam)
    
  }
}

gam.best.fit = glm(donr~reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                     avhv + incm + plow + npro + tgif + poly(lgif,4) + rgif + 
                     poly(tdon, 4) + poly(tlag, 2) + poly(agif, 3),
                   data=data.train.std.c, family=binomial("logit"))

#summary(gam.best.fit)

gam.best.preds = predict(gam.best.fit, data.valid.std.c, type="response")
profit.gam = cumsum(14.5*c.valid[order(gam.best.preds, decreasing=T)]-2)
plot(profit.gam) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.gam) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.gam)) # report number of mailings and maximum profit

cutoff.gam <- sort(gam.best.preds, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.gam <- ifelse(gam.best.preds>cutoff.gam, 1, 0) # mail to everyone above the cutoff
table(chat.valid.gam, c.valid) # classification table

#Support Vector Machine
#WARNING: This code will take a while to execute
#svm.tune = tune(svm, donr~.-damt, data=data.train, ranges = list(cost=c(0.001, 0.01, 0.1,
#                                                     1, 5, 10), 
#                                                kernel = c('linear'), 
#                                                gamma = c(0.5, 1, 2, 3, 4)))
#Cost = 0.01, Gamma = 0.5
#svm.best = svm(donr~.-damt, data=data.train, cost=0.01, gamma=0.5, kernel = "linear")
#svm.preds = predict(svm.best, data.val)

#table(pred = svm.preds, true = data.val$donr)

#mean(svm.preds==data.val$donr)

#WARNING: This code will take a while to execute
set.seed(1)
svm.tune2 = tune(svm, donr~.-damt, data=data.train, ranges = list(cost=c(9, 8, 10, 11, 12), 
                                                                 kernel = c('radial'), 
                                                                 gamma = c(0.011, 0.015, 0.013)))
#Cost = 6, gamma = 0.01, kernel = radial 10, 0.011
#svm.best2 = svm.tune2$best.model
svm.best2 = svm(donr~.-damt, data=data.train, probability=TRUE, cost = 10, kernel = 'radial', gamma = 0.011)
summary(svm.best2)
svm.preds2 = predict(svm.best2, data.val, probability = TRUE)

table(pred = svm.preds2, true = data.val$donr)

mean(svm.preds2==data.val$donr)

profit.log1 <- cumsum(14.5*c.valid[order(attr(svm.preds2, "probabilities")[,2], decreasing=T)]-2)
plot(profit.log1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.log1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.log1)) # report number of mailings and maximum profit

#Naive Bayes
nb.fit = naiveBayes(donr~.-damt, data=data.train, probability=TRUE)

nb.preds = predict(nb.fit, data.val[,!colnames(data.val) %in% c('donr','damt')], probability=TRUE)

table(pred = nb.preds, true = data.val$donr)

mean(nb.preds == data.val$donr)

profit.log1 <- cumsum(14.5*c.valid[order(nb.preds, decreasing=T)]-2)
plot(profit.log1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.log1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.log1)) # report number of mailings and maximum profit

#Decision Tree
set.seed(3)
tree.donor = tree(donr~.-damt, data=data.train)
cv.donor = cv.tree(tree.donor, FUN=prune.misclass)
names(cv.donor)

cv.donor

prune.donor = prune.misclass(tree.donor, best=15)

tree.preds = predict(prune.donor, newdata=data.val, type='vector')

table(pred = tree.preds, true = data.val$donr)

profit.log1 <- cumsum(14.5*c.valid[order(tree.preds[,2], decreasing=T)]-2)
plot(profit.log1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.log1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.log1)) # report number of mailings and maximum profit

#Bagging
set.seed(1)
bag.fit = randomForest(donr~.-damt, data=data.train, mtry=20, importance=TRUE, probability=TRUE)
bag.preds = predict(bag.fit, newdata=data.val, type='prob')

table(pred = bag.preds, true = data.val$donr)

mean(bag.preds==data.val$donr)

profit.log1 <- cumsum(14.5*c.valid[order(bag.preds[,2], decreasing=T)]-2)
plot(profit.log1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.log1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.log1)) # report number of mailings and maximum profit

#RandomForest
set.seed(1)
rf.fit = randomForest(donr~.-damt, data=data.train, mtry=8, probability=TRUE, importance=TRUE)
rf.preds = predict(rf.fit, newdata=data.val, type="prob")
rf.preds.c = predict(rf.fit, newdata=data.val)
table(predict = rf.preds.c, true = data.val$donr)

mean(rf.preds==data.val$donr)
profit.log1 <- cumsum(14.5*c.valid[order(rf.preds[,2], decreasing=T)]-2)
plot(profit.log1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.log1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.log1)) # report number of mailings and maximum profit


registerDoMC(4) #Set the number of cores here

foreach(i=1:20)%dopar%{
  set.seed(1)
  fit = randomForest(donr~.-damt, data=data.train, mtry=i, importance=TRUE)
  preds = predict(fit, newdata=data.val, type='prob')
  
  print(i)
  print(mean(preds==data.val$donr))
  
  profit.log1 <- cumsum(14.5*c.valid[order(preds[,2], decreasing=T)]-2)
  print(max(profit.log1))

}

#Gradient Boosting
set.seed(123)
gbm.fit = gbm(c.train~.-donr-damt,data=data.train, 
                                  distribution = 'bernoulli',
                                  n.trees = 5500,
                                  shrinkage = 0.015,
                                  interaction.depth = 1,
                                  bag.fraction = 0.7,
                                  n.minobsinnode = 42,verbose=F)
gbm.preds = predict(gbm.fit, newdata=data.val, n.trees = 5200, type='response')
summary(gbm.fit)
profit.gbm <- cumsum(14.5*c.valid[order(gbm.preds, decreasing=T)]-2)
plot(profit.gbm) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.gbm) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.gbm)) # report number of mailings and maximum profit

gbm.preds[gbm.preds>=0.5] = 1
gbm.preds[gbm.preds<0.5] = 0

table(pred = gbm.preds, true = c.valid)
mean(gbm.preds==c.valid)

cutoff.gbm <- sort(gbm.preds, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.gbm <- ifelse(gbm.preds>cutoff.gbm, 1, 0) # mail to everyone above the cutoff
table(chat.valid.gbm, c.valid) # classification table

n.mail.valid <- which.max(profit.gbm)
tr.rate <- .1 # typical response rate is .1
vr.rate <- .5 # whereas validation response rate is .5
adj.test.1 <- (n.mail.valid/n.valid.c)/(vr.rate/tr.rate) # adjustment for mail yes
adj.test.0 <- ((n.valid.c-n.mail.valid)/n.valid.c)/((1-vr.rate)/(1-tr.rate)) # adjustment for mail no
adj.test <- adj.test.1/(adj.test.1+adj.test.0) # scale into a proportion
n.mail.test <- round(n.test*adj.test, 0) # calculate number of mailings for test set

cutoff.test <- sort(post.test, decreasing=T)[n.mail.test+1] # set cutoff based on n.mail.test
chat.test <- ifelse(post.test>cutoff.test, 1, 0) # mail to everyone above the cutoff
table(chat.test)

#Find optimal shrinkage
#WARNING: This section will take a while to run
shrinks = c(0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02)

foreach(i=shrinks)%dopar%{
  set.seed(123)
  gbm.fit = gbm(c.train~.-donr-damt,data=data.train, 
                distribution = 'bernoulli',
                n.trees = 5000,
                shrinkage = i,
                n.minobsinnode = 43, verbose=F)
  gbm.preds = predict(gbm.fit, newdata=data.val, n.trees = 5000, type='response')
  
  profit.log1 <- cumsum(14.5*c.valid[order(gbm.preds, decreasing=T)]-2)
  print(i)
  print(shrinks[i])
  print(max(profit.log1))
}

#Find optimal trees
#WARNING: This section will take a long time to run
set.seed(123)
gbm.fit = gbm(c.train~.-donr-damt,data=data.train, 
              distribution = 'bernoulli',
              n.trees = 10000,
              shrinkage = 0.015,
              n.minobsinnode = 48, verbose=F)

profits = c()
start = proc.time()
for (i in 1:10000){
  if(i %% 1000 == 0){
    print(i)
    print(proc.time() - start)
    start = proc.time()
  }
  gbm.preds = predict(gbm.fit, newdata=data.val, n.trees = i, type='response')
  profit.log1 <- cumsum(14.5*c.valid[order(gbm.preds, decreasing=T)]-2)
  profits[i] = max(profit.log1)
}

#Find optimal minobs
minobs = c(40, 41, 42, 43, 44, 45)
profits = c()

foreach(i=minobs)%dopar%{
set.seed(123)
gbm.fit = gbm(c.train~.-donr-damt,data=data.train, 
              distribution = 'bernoulli',
              n.trees = 5500,
              shrinkage = 0.015,
              interaction.depth = 1,
              bag.fraction=0.7,
              n.minobsinnode = i, verbose=F)
gbm.preds = predict(gbm.fit, newdata=data.val, n.trees = 5200, type='response')

profit.log1 <- cumsum(14.5*c.valid[order(gbm.preds, decreasing=T)]-2)
profits = max(profit.log1)
}


#Least Squares Regression
lm.fit1 = lm(damt~., data=data.train.std.y)
#summary(lm.fit)
lm.preds1 = predict(lm.fit1, newdata=data.valid.std.y)
mean((reg.data.val$damt - lm.preds1)^2)

lm.fit2 = lm(damt~.-inca, data=data.train.std.y)
#summary(lm.fit)
lm.preds2 = predict(lm.fit2, newdata=data.valid.std.y)
mean((reg.data.val$damt - lm.preds2)^2)

lm.fit3 = lm(damt~.-inca-wrat, data=data.train.std.y)
#summary(lm.fit)
lm.preds3 = predict(lm.fit3, newdata=data.valid.std.y)
mean((reg.data.val$damt - lm.preds3)^2)

#Use this model
lm.fit4 = lm(damt~.-inca-wrat-avhv, data=data.train.std.y)
summary(lm.fit4)
lm.preds4 = predict(lm.fit4, newdata=data.valid.std.y)
mean((reg.data.val$damt - lm.preds4)^2)
plot(reg.data.val$damt - lm.preds4)

lm.fit5 = lm(damt~.-inca-wrat-avhv-tlag, data=data.train.std.y)
#summary(lm.fit)
lm.preds5 = predict(lm.fit5, newdata=data.valid.std.y)
mean((reg.data.val$damt - lm.preds5)^2)

plot(reg.data.val$damt - lm.preds5)

ncvTest(lm.fit)
vif(lm.fit)
sqrt(vif(lm.fit)) > 2
qqPlot(lm.fit)

#Best Subset
library(leaps)

predict.regsubsets = function(object, newdata, id, ...){
  form = as.formula(object$call[[2]])
  mat = model.matrix(form, newdata)
  coefi = coef(object, id=id)
  xvars = names(coefi)
  mat[,xvars]%*%coefi
}

k = 10
set.seed(1)
folds = sample(1:k, nrow(data.train.std.y), replace=TRUE)

cv.errors = matrix(NA, k, 20, dimnames = list(NULL, paste(1:20)))

for(j in 1:k){
  best.fit = regsubsets(damt~.,data=data.train.std.y[folds!=j,], nvmax=20)
  for(i in 1:20){
    pred = predict(best.fit, data.train.std.y[folds==j,], id=i)
    cv.errors[j,i] = mean((data.train.std.y$damt[folds==j] - pred)^2)
  }
}

mean.cv.errors = apply(cv.errors, 2, mean)
#outputs 13 as the min

reg.best = regsubsets(damt~., data=data.train.std.y, nvmax=20)
coef(reg.best, 13)

reg.best.lm = lm(damt~ reg3 + reg4 + home + 
                   chld + hinc + genf + incm + plow + 
                   npro + rgif + tdon + agif, data=data.train.std.y)

subset.pred = predict(reg.best.lm, newdata=data.valid.std.y)
mean((y.valid - subset.pred)^2)

plot(y.valid - subset.pred)

#Gradient Boosting Regression
set.seed(123)
gbm.reg.fit = gbm(damt~.,data=data.train.std.y, 
                distribution = "gaussian",
              n.trees = 6000,
              shrinkage = 0.01,
              interaction.depth = 1,
              bag.fraction = 0.8,
              n.minobsinnode = 4, verbose=F)
gbm.reg.preds = predict(gbm.reg.fit, newdata=data.valid.std.y, n.trees = 5500)
summary(gbm.reg.fit)

mean((y.valid - gbm.reg.preds)^2)
plot(y.valid - gbm.reg.preds)

mpe = c()
start = proc.time()
for (i in 1:10000){
  if(i %% 1000 == 0){
    print(i)
    print(proc.time() - start)
    start = proc.time()
  }
  gbm.preds = predict(gbm.reg.fit, newdata=data.valid.std.y, n.trees = i)
  mpe[i] = mean((y.valid - gbm.preds)^2)
}

plot(mpe)

mpe2 = c()
#started with range of 40:60, keep adjusting until find the min
for(i in 1:20){
  set.seed(123)
  gbm.reg.fit = gbm(damt~.,data=data.train.std.y, 
                    distribution = "gaussian",
                    n.trees = 5000,
                    shrinkage = 0.01,
                    interaction.depth = 1,
                    n.minobsinnode = i, verbose=F)
  gbm.reg.preds = predict(gbm.reg.fit, newdata=data.valid.std.y, n.trees = 4250)
  
  mpe2[i] = mean((y.valid - gbm.reg.preds)^2)
  #print(i)
}

min(mpe2)
which.min(mpe2)
#Random Forest Regression
foreach(i=1:20)%dopar%{
  set.seed(1)
  fit = randomForest(damt~., data=data.train.std.y, mtry=i, importance=TRUE)
  preds = predict(fit, newdata=data.valid.std.y)
  
  mpe = mean((y.valid - preds)^2)
  print(mpe)
  
}

set.seed(1)
rf.reg.fit = randomForest(damt~., data=data.train.std.y, mtry=5, importance=TRUE)
sort(importance(rf.reg.fit)[,2], decreasing = T)
rf.reg.preds = predict(rf.reg.fit, newdata=data.valid.std.y)

mean((y.valid - rf.reg.preds)^2)
