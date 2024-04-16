
vacc <- read.table("vacc_BCG.txt", header = TRUE, check.names = FALSE) #check.names fixes the column names (removes X's)

any(duplicated(vacc$iso3)) # no duplicates in country shortcuts
rownames(vacc) # so far only numbers, can be overwritten
rownames(vacc) <- vacc$iso3 # overwrite 
vacc$iso3 <- NULL #get rid of column

plot(x = as.numeric(colnames(vacc)),y = unlist(vacc["AFG",]), 
     type = "l", lwd = 4.3, col = "orange", las = 1, ylab = "percentage of children vaccinated", xlab = "years") # unlist puts it into a vector  


