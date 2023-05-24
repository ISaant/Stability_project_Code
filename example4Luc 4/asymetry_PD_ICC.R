# read data and compute data for demographic table
data=read.csv('~/Documents/PD_fingerprinting/QPN_demo_compiled.csv', stringsAsFactors = FALSE)

data %>% group_by(Group) %>% summarise(m_age= mean(Age, na.rm=TRUE), sd_age= sd(Age, na.rm=TRUE))
table(data$Group, data$Sex)
table(data$Group, data$Handedness)

target= read.csv('~/Documents/PD_fingerprinting/new_analysis_Nov2022/NEW_aperiodic_corrected_target.csv' )

database= read.csv('~/Documents/PD_fingerprinting/new_analysis_Nov2022/NEW_aperiodic_corrected_database.csv' )

target=target[,-1]
database=database[,-1]


z_target= t(scale(t(target[data$Group== "Parkinson",])))
z_database= t(scale(t(database[data$Group== "Parkinson",])))
icc = c()


n = 79
k = 2
df_b = n-1
df_w = n*(k-1)


for (i_edge in 1:length(target)-1){
  x= data.frame(unlist(z_target[,i_edge]),unlist(z_database[,i_edge]) )
  x_w_mean =  rowMeans(x)
  x_g_mean = mean(unlist(x))
  ss_t = sum(((x - x_g_mean)^2))
  ss_w = sum((x - (x_w_mean))^2)
  ss_b = ss_t - ss_w
  ms_b = ss_b / df_b
  ms_w = ss_w / df_w
  icc[i_edge] = (ms_b - ms_w) / (ms_b + ((k-1)*ms_w))
  
  
}

icc_mat_PD <- matrix(icc, nrow = 68, byrow = TRUE)

X=data.frame(theta= rowMeans(icc_mat_PD[,7:18]),alpha= rowMeans(icc_mat_PD[,19:33]),beta= rowMeans(icc_mat_PD[,34:84]), gamma= rowMeans(icc_mat_PD[,84:115]))

df=read.csv('~/Documents/PD_fingerprinting/new_analysis_Nov2022/2023correctedspectra_ICC_PD.csv', header = FALSE, stringsAsFactors = FALSE)
df=df[,-1]
Xo=data.frame(theta= rowMeans(df[,7:18]),alpha= rowMeans(df[,19:33]),beta= rowMeans(df[,34:84]), gamma= rowMeans(df[,84:115]))

cor.test(rowMeans(X), rowMeans(Xo))




z_target= t(scale(t(target[data$Group== "Control",])))
z_database= t(scale(t(database[data$Group== "Control",])))
icc = c()

n = 54
k = 2
df_b = n-1
df_w = n*(k-1)


for (i_edge in 1:length(target)-1){
  x= data.frame(unlist(z_target[,i_edge]),unlist(z_database[,i_edge]) )
  x_w_mean =  rowMeans(x)
  x_g_mean = mean(unlist(x))
  ss_t = sum(((x - x_g_mean)^2))
  ss_w = sum((x - (x_w_mean))^2)
  ss_b = ss_t - ss_w
  ms_b = ss_b / df_b
  ms_w = ss_w / df_w
  icc[i_edge] = (ms_b - ms_w) / (ms_b + ((k-1)*ms_w))
  
  
}

icc_mat_CTL <- matrix(icc, nrow = 68, byrow = TRUE)

XCTL=data.frame(theta= rowMeans(icc_mat_CTL[,7:18]),alpha= rowMeans(icc_mat_CTL[,19:33]),beta= rowMeans(icc_mat_CTL[,34:84]), gamma= rowMeans(icc_mat_CTL[,84:115]))




index= data$Asymetrie== "Droite"
index[is.na(index)]= FALSE
z_target= t(scale(t(target[index,])))
z_database= t(scale(t(database[index,])))
icc = c()


n = 32
k = 2
df_b = n-1
df_w = n*(k-1)


for (i_edge in 1:length(target)-1){
  x= data.frame(unlist(z_target[,i_edge]),unlist(z_database[,i_edge]) )
  x_w_mean =  rowMeans(x)
  x_g_mean = mean(unlist(x))
  ss_t = sum(((x - x_g_mean)^2))
  ss_w = sum((x - (x_w_mean))^2)
  ss_b = ss_t - ss_w
  ms_b = ss_b / df_b
  ms_w = ss_w / df_w
  icc[i_edge] = (ms_b - ms_w) / (ms_b + ((k-1)*ms_w))
  
  
}

icc_mat_Droite <- matrix(icc, nrow = 68, byrow = TRUE)


index= data$Asymetrie== "Gauche"
index[is.na(index)]= FALSE
z_target= t(scale(t(target[index,])))
z_database= t(scale(t(database[index,])))
icc = c()


n = 34
k = 2
df_b = n-1
df_w = n*(k-1)


for (i_edge in 1:length(target)-1){
  x= data.frame(unlist(z_target[,i_edge]),unlist(z_database[,i_edge]) )
  x_w_mean =  rowMeans(x)
  x_g_mean = mean(unlist(x))
  ss_t = sum(((x - x_g_mean)^2))
  ss_w = sum((x - (x_w_mean))^2)
  ss_b = ss_t - ss_w
  ms_b = ss_b / df_b
  ms_w = ss_w / df_w
  icc[i_edge] = (ms_b - ms_w) / (ms_b + ((k-1)*ms_w))
  
  
}

icc_mat_Gauche <- matrix(icc, nrow = 68, byrow = TRUE)

XGauche=data.frame(theta= rowMeans(icc_mat_Gauche[,7:18]),alpha= rowMeans(icc_mat_Gauche[,19:33]),beta= rowMeans(icc_mat_Gauche[,34:84]), gamma= rowMeans(icc_mat_Gauche[,84:115]))

XDroite=data.frame(theta= rowMeans(icc_mat_Droite[,7:18]),alpha= rowMeans(icc_mat_Droite[,19:33]),beta= rowMeans(icc_mat_Droite[,34:84]), gamma= rowMeans(icc_mat_Droite[,84:115]))


# now check how change in cortical thickness relates to ICC and differentiability 
#### diff in ICC between PD and CTLS
df=read.csv('~/Documents/PD_fingerprinting/new_analysis_Nov2022/2023correctedspectra_ICC_controls.csv', header = FALSE, stringsAsFactors = FALSE)
colnames(df)[1]='region'
df$hemi =''
df$hemi[seq(2,68,2)] ='right'
df$hemi[seq(1,67,2)] ='left'

someData= tidyr::tibble(df$region, rowMeans(X), df$hemi)
colnames(someData)[2]='ICC'
colnames(someData)[1]='region'
colnames(someData)[3]='hemi'


ggplot(someData) +
  geom_brain(atlas = dk, 
             position = position_brain(hemi ~ side),
             aes(fill = `ICC`)) +
  #scale_fill_distiller(palette = "RdBu"
  #                     , limits = c(0.4, 0.9))+
  viridis::scale_fill_viridis(option = 'magma')+
  #scale_fill_gradient2(low="white", mid="#EC352F", high="#6C1917", 
  #                     midpoint=0.6,   
  #                     limits=c(0.4, 0.9 ))+
  theme_void() 


someData= tidyr::tibble(df$region, rowMeans(XDroite), df$hemi)
colnames(someData)[2]='ICC'
colnames(someData)[1]='region'
colnames(someData)[3]='hemi'


ggplot(someData) +
  geom_brain(atlas = dk, 
             position = position_brain(hemi ~ side),
             aes(fill = `ICC`)) +
  #scale_fill_distiller(palette = "RdBu"
  #                     , limits = c(0.4, 0.9))+
  viridis::scale_fill_viridis(option = 'magma')+
  #scale_fill_gradient2(low="white", mid="#EC352F", high="#6C1917", 
  #                     midpoint=0.6,   
  #                     limits=c(0.4, 0.9 ))+
  theme_void() 


someData= tidyr::tibble(df$region, rowMeans(XGauche)-rowMeans(XCTL), df$hemi)
colnames(someData)[2]='ICC'
colnames(someData)[1]='region'
colnames(someData)[3]='hemi'

someData$ICC[someData$ICC > 0.3]=0.3
someData$ICC[someData$ICC < -0.3]= -0.3

ggplot(someData) +
  geom_brain(atlas = dk, 
             position = position_brain(hemi ~ side),
             aes(fill = `ICC`)) +
  scale_fill_gradient2(low = "#67309A", mid = "#FCF7F4", high = "#EC7A48", midpoint = 0.0, limits=c(-0.3, 0.3 )) +
  theme_void() 


someData= tidyr::tibble(df$region, rowMeans(XDroite)-rowMeans(XCTL), df$hemi)
colnames(someData)[2]='ICC'
colnames(someData)[1]='region'
colnames(someData)[3]='hemi'

someData$ICC[someData$ICC > 0.3]=0.3
someData$ICC[someData$ICC < -0.3]= -0.3


ggplot(someData) +
  geom_brain(atlas = dk, 
             position = position_brain(hemi ~ side),
             aes(fill = `ICC`)) +
  scale_fill_gradient2(low = "#67309A", mid = "#FCF7F4", high = "#EC7A48", midpoint = 0.0, limits=c(-0.3, 0.3 )) +
  theme_void() 



################### REMOVE OUTLIERS

data_temp=data[-c(20,29),]
target_temp=target[-c(20,29),]
database_temp=database[-c(20,29),]
z_target= t(scale(t(target_temp[data_temp$Group== "Control",])))
z_database= t(scale(t(database_temp[data_temp$Group== "Control",])))
icc = c()

n = 52
k = 2
df_b = n-1
df_w = n*(k-1)


for (i_edge in 1:length(target)-1){
  x= data.frame(unlist(z_target[,i_edge]),unlist(z_database[,i_edge]) )
  x_w_mean =  rowMeans(x)
  x_g_mean = mean(unlist(x))
  ss_t = sum(((x - x_g_mean)^2))
  ss_w = sum((x - (x_w_mean))^2)
  ss_b = ss_t - ss_w
  ms_b = ss_b / df_b
  ms_w = ss_w / df_w
  icc[i_edge] = (ms_b - ms_w) / (ms_b + ((k-1)*ms_w))
  
  
}

icc_mat_CTL_test <- matrix(icc, nrow = 68, byrow = TRUE)

XCTL_test=data.frame(theta= rowMeans(icc_mat_CTL_test[,7:18]),alpha= rowMeans(icc_mat_CTL_test[,19:33]),beta= rowMeans(icc_mat_CTL_test[,34:84]), gamma= rowMeans(icc_mat_CTL_test[,84:115]))

cor.test(XCTL[,1], XCTL_test[,1])

cor.test(rowMeans(XCTL), rowMeans(XCTL_test)) # 0.93

# now check how change in cortical thickness relates to ICC and differentiability 
#### diff in ICC between PD and CTLS
df=read.csv('~/Documents/PD_fingerprinting/new_analysis_Nov2022/2023correctedspectra_ICC_controls.csv', header = FALSE, stringsAsFactors = FALSE)
colnames(df)[1]='region'
df$hemi =''
df$hemi[seq(2,68,2)] ='right'
df$hemi[seq(1,67,2)] ='left'

someData= tidyr::tibble(df$region, rowMeans(XCTL_test), df$hemi)
colnames(someData)[2]='ICC'
colnames(someData)[1]='region'
colnames(someData)[3]='hemi'


ggplot(someData) +
  geom_brain(atlas = dk, 
             position = position_brain(hemi ~ side),
             aes(fill = `ICC`)) +
  #scale_fill_distiller(palette = "RdBu"
  #                     , limits = c(0.4, 0.9))+
  viridis::scale_fill_viridis(option = 'magma', limits=c(0.4, 1 ))+
  #scale_fill_gradient2(low="white", mid="#EC352F", high="#6C1917", 
  #                     midpoint=0.6,   
  #                     limits=c(0.4, 0.9 ))+
  theme_void() 



difference= icc_mat_PD- icc_mat_CTL
DIFFOrig=data.frame(theta= rowMeans(difference[,7:18]),alpha= rowMeans(difference[,19:33]),beta= rowMeans(difference[,34:84]), gamma= rowMeans(difference[,84:115]))

difference= icc_mat_PD- icc_mat_CTL_test
DIFF=data.frame(theta= rowMeans(difference[,7:18]),alpha= rowMeans(difference[,19:33]),beta= rowMeans(difference[,34:84]), gamma= rowMeans(difference[,84:115]))

cor.test(rowMeans(DIFF), rowMeans(DIFFOrig)) # 0.96


# now check how change in cortical thickness relates to ICC and differentiability 
#### diff in ICC between PD and CTLS
df=read.csv('~/Documents/PD_fingerprinting/new_analysis_Nov2022/2023correctedspectra_ICC_controls.csv', header = FALSE, stringsAsFactors = FALSE)
colnames(df)[1]='region'
df$hemi =''
df$hemi[seq(2,68,2)] ='right'
df$hemi[seq(1,67,2)] ='left'

someData= tidyr::tibble(df$region, rowMeans(DIFF), df$hemi)
colnames(someData)[2]='ICC'
colnames(someData)[1]='region'
colnames(someData)[3]='hemi'

someData$ICC[someData$ICC >0.3] =0.3
someData$ICC[someData$ICC < -0.3] = -0.3

ggplot(someData) +
  geom_brain(atlas = dk, 
             position = position_brain(hemi ~ side),
             aes(fill = `ICC`)) +
  scale_fill_gradient2(low = "#67309A", mid = "#FCF7F4", high = "#EC7A48", midpoint = 0.0, limits=c(-0.3, 0.3 )) +
  theme_void() 


#
SVM=read.csv( '~/Documents/PD_fingerprinting/Results_from_SVM_class_of_HY_aperiodic_corrected_80_20.csv')
SVM=SVM[,-1]
cor.test(rowMeans(DIFF), colMeans(SVM))

permuted_index= read.csv('~/Documents/CAMCAN_outputs/neuromaps/permuted_indexes_of_dk_atlas.csv')
permuted_index= permuted_index[,-1]
permuted_index=permuted_index+1


#gradient offset
orig=cor.test(rowMeans(DIFF), colMeans(SVM))
permuted_corr=c()
for (i in 1:1000){
  
  cor_temp=cor.test(rowMeans(DIFF), colMeans(SVM)[permuted_index[,i]])
  permuted_corr= c(permuted_corr, cor_temp$estimate)
  
}

sum(orig$estimate < permuted_corr)/1000 # 0.003


### func grad
neuromaps= read.csv('~/Documents/PD_fingerprinting/neuromaps/receptor_data_scale033_full.csv')
cor.test(rowMeans(DIFF), neuromaps$func_grad_01)

cor.test(rowMeans(DIFF), neuromaps$func_grad_01)

cor.test(rowMeans(DIFF), neuromaps$NET)
cor.test(rowMeans(DIFF), neuromaps$MOR)
cor.test(rowMeans(DIFF), neuromaps$CB1)
cor.test(rowMeans(DIFF), neuromaps$X5HT4)
cor.test(rowMeans(DIFF), neuromaps$X5HT2a)




atlas_dk= read.csv('~/Documents/CAMCAN_outputs/DK_atlas_mni_coord.csv')
atlas_dk[10:12]= scale(atlas_dk[10:12])

atlas_dk$ICC= rowMeans(DIFFOrig)

lm0= lm(ICC ~ 1, atlas_dk)
lm1= lm(ICC ~ x.mni + y.mni + z.mni, atlas_dk)

sjPlot::tab_model(lm1)