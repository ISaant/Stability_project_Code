setwd('~/Documents/Doctorado CIC/Internship/Sylvain/Stability-project/Stability_project_Code/example4Luc 4/')
cbbPalette= c("#DBF132", "#15A705",'#4F89E0',"#D81B99",'#4D3D87','#115D80', '#FF0000')

df=read.csv('./dka_data.csv', header = TRUE, stringsAsFactors = FALSE)
colnames(df)[1]='region'

df$hemi =''
df$hemi[seq(2,68,2)] ='right'
df$hemi[seq(1,67,2)] ='left'

colnames(df)[2]='decoding_corr'
colnames(df)[4]='YEO'
df$YEO_con= as.numeric(plyr::mapvalues(df$YEO, unique(df$YEO), 1:7))


ggplot(df) +
  geom_brain(atlas = dk, 
             position = position_brain(hemi ~ side),
             aes(fill = 'decoding_corr')) + scale_fill_manual(values=cbbPalette) + scale_colour_manual(values=cbbPalette) +
  theme_void() 

ggsave('~/Documents/Doctorado CIC/Internship/Sylvain/Stability-project/Stability_project_Code/example4Luc 4/decoding_of_age_80_20.pdf', device = "pdf")


ggplot(df, aes(YEO, decoding_corr, fill=YEO)) +
  ggdist::stat_halfeye(adjust = .5, width = .1, .width = 0, justification = -.5, alpha=0.5, point_alpha= 0) +
  geom_boxplot(width = .3, outlier.shape = NA, colour= '#888888') + ggpubr::theme_classic2() + scale_fill_manual(values=cbbPalette) #+ facet_wrap(~ ind)

# df=read.csv('dka_data.csv', header = TRUE, stringsAsFactors = FALSE)
# colnames(df)[2]='decoding corr'
# df$`decoding corr`[df$`decoding corr`<0.5]=0.5
# df$`decoding corr`[df$`decoding corr`>1.0]=1.0
# 
# ggplot(df) +
#   geom_brain(atlas = dk, 
#              position = position_brain(hemi ~ side),
#              aes(fill = `decoding corr`)) +
#   #scale_fill_distiller(palette = "RdBu"
#   #                     , limits = c(0.4, 0.9))+
#   viridis::scale_fill_viridis(option = 'viridis', limits = c(0.5, 1.0))+
#   #scale_fill_gradient2(low="white", mid="#EC352F", high="#6C1917", 
#   #                     midpoint=0.6,   
#   #                     limits=c(0.4, 0.9 ))+
#   theme_void() 
# 
# ggsave('~/Documents/CAMCAN_outputs/figures/decoding_of_age_80_20.pdf', device = "pdf")
# 
# 
# ggplot(df) +
#   geom_brain(atlas = dk, 
#              position = position_brain(hemi ~ side),
#              aes(fill = alpha)) +
#   viridis::scale_fill_viridis(option = 'viridis')+
#   theme_void() 
# 
# ggplot(df) +
#   geom_brain(atlas = dk, 
#              position = position_brain(hemi ~ side),
#              aes(fill = theta)) +
#   viridis::scale_fill_viridis(option = 'viridis')+
#   theme_void() 
# 
# ggplot(df) +
#   geom_brain(atlas = dk, 
#              position = position_brain(hemi ~ side),
#              aes(fill = beta)) +
#   viridis::scale_fill_viridis(option = 'viridis')+
#   theme_void() 