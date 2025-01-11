##
##
##   Project 1
##
##   CFRM503, University of Washington
##
##   Steve Murray, April 29, 2024
##

# clear environment
rm(list = ls())

# libraries and packages
library(quantmod)
library(tidyverse)
library(moments)
library(CVXR)
#library(glpkAPI)


# Get data
from = as.Date("01/01/2005", format = "%m/%d/%Y")
to = as.Date("04/01/2024", format = "%m/%d/%Y")

yahooSymbols_df = data.frame(symbols = c("IWB", "IWM",
                                         "EFA", "EEM",
                                         "VNQ", "LQD","SHY"),
                             label = c("Large Cap US","Small Cap US",
                                       "Dev. Mkts", "Emer Mkts",
                                       "Global REIT", "Corp. Bonds","Short Tsy"))


invisible(getSymbols(yahooSymbols_df$symbols, src = "yahoo", from = from , to = to))

yahooReturnData_xts <- do.call("cbind",lapply(yahooSymbols_df$symbols, FUN = function(symb){
  temp <- eval(parse(text = symb))
  #  colnames(temp) <-  paste0(symb,c(".Open",".High",".Low",".Close",".Volume",".Adjusted"))
  temp <- temp[,6]
  #  colnames(temp) <- c(paste0(symb,""))
}))
colnames(yahooReturnData_xts) <- gsub(".Adjusted","",colnames(yahooReturnData_xts))

temp <- log(yahooReturnData_xts/stats::lag(yahooReturnData_xts))

temp <- temp[-1,]

financialDataReturns_xts <- temp

dim(financialDataReturns_xts)


#
# Aggregate returns to 5, 21, 62 and 252 day intervals
#

return_list = list()
return_list_df = list()
for(numDays in c(1, 5, 21, 62, 252)){
  return_xts = rollapply(financialDataReturns_xts,
                          width = numDays, 
                          sum,
                          by = numDays,
                          align = "right")
  item_name =  paste0("days_",numDays)
# add non-NA rows of returns_xts to list
  return_list[[item_name]] = return_xts[rowSums((is.na(return_xts))) == 0,]
  temp_df <- as.data.frame(return_list[[item_name]])
  temp_df <- temp_df %>% rownames_to_column("Date")
  return_list_df[[item_name]] = temp_df
}

# check number of observations for each time period length
lapply(return_list, dim)

# write observations to Excel file
library(writexl)
folderName <- "C:/Users/arize/Documents/CFRM/CFRM503/2024/Assignments/Project 1"
fileName <- "Project1Data.xlsx"
write_xlsx(return_list_df, file.path(folderName, fileName),
           col_names = TRUE)


#
# calculated statistics
#     mean, standard deviation, skewness, kurtosis
#     10th, 25th, 50th, 75th, 90th percentiles
#
stats_df = data.frame()
correlation_list = list()
for(numDays in c(1,5,21,62,252)){
  rets <- return_list[[paste0("days_",numDays)]]
  
  # calculate means
  temp <- as.data.frame(t(apply(rets, 2, mean)))
  temp <- temp %>% mutate(numDays = numDays, statistic = "mean")
  stats_df <- rbind(stats_df, temp)
  
  # calculate standard deviations
  temp <- as.data.frame(t(apply(rets, 2, sd)))
  temp <- temp %>% mutate(numDays = numDays, statistic = "stdev")
  stats_df <- rbind(stats_df, temp)
  
  # calculate skewness
  temp <- as.data.frame(t(apply(rets, 2, skewness)))
  temp <- temp %>% mutate(numDays = numDays, statistic = "skew")
  stats_df <- rbind(stats_df, temp)
  
  # calculate kurtosis
  temp <- as.data.frame(t(apply(rets, 2, kurtosis)))
  temp <- temp %>% mutate(numDays = numDays, statistic = "kurtosis")
  stats_df <- rbind(stats_df, temp)
  
  # calculate quantiles
  for(quant in c(0.1, 0.25, 0.5, 0.75, 0.9)){
    temp <- as.data.frame(t(apply(rets, 2, quantile, probs = quant)))
    temp <- temp %>% mutate(numDays = numDays, statistic = paste0("quantile_",quant))
    stats_df <- rbind(stats_df, temp)
  }
  
  # calculate correlations
  correlation_list[[paste0("days_",numDays)]] = cor(rets)
}


#
# arrange data frame by period length
# and in order mean, standard deviation, skewness, kurtosis
# and quantiles
#

stat_order <- c("mean","stdev","skew","kurtosis",
                "quantile_0.1", "quantile_0.25",
                "quantile_0.5", "quantile_0.75",
                "quantile_0.9")
stats_df <- stats_df %>% 
              mutate(statistic = factor(statistic, 
                                       levels = stat_order)) %>%
              arrange(statistic, numDays)

