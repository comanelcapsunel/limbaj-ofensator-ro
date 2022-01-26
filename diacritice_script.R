getwd()
setwd('D:\\Master\\DM An 2\\DM21-S1\\2 - Prelucrarea Statistica a Datelor\\Proiect Seminar TEAM 4\\eliminare diacritice')

#import date
library("readxl")
romanian_train_texts_classified <- read_excel('D:\\Master\\DM An 2\\DM21-S1\\2 - Prelucrarea Statistica a Datelor\\Proiect Seminar TEAM 4\\eliminare diacritice\\romanian_train_texts_classified_og.xlsx')
romanian_bad_words_list <- read_excel('D:\\Master\\DM An 2\\DM21-S1\\2 - Prelucrarea Statistica a Datelor\\Proiect Seminar TEAM 4\\eliminare diacritice\\romanian_bad_words_list_og.xlsx')


#install.packages('tidyverse')
library('tidyr')
romanian_train_texts_classified$text <- stringi::stri_trans_general(romanian_train_texts_classified$text, "Latin-ASCII")
romanian_bad_words_list$ro_bad_words <- stringi::stri_trans_general(romanian_bad_words_list$ro_bad_words, "Latin-ASCII")

write.csv(romanian_train_texts_classified,"D:\\Master\\DM An 2\\DM21-S1\\2 - Prelucrarea Statistica a Datelor\\Proiect Seminar TEAM 4\\eliminare diacritice\\romanian_train_texts_classified.csv", row.names = FALSE)
write.csv(romanian_bad_words_list,"D:\\Master\\DM An 2\\DM21-S1\\2 - Prelucrarea Statistica a Datelor\\Proiect Seminar TEAM 4\\eliminare diacritice\\romanian_bad_words_list.csv", row.names = FALSE)
