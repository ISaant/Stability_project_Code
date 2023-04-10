# ==============================================================================
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def ACP (DataFrame):
    print('----------------------')
    print('Media de cada variable')
    print('----------------------')
    print(DataFrame.mean(axis=0))
    
    
    print('-------------------------')
    print('Varianza de cada variable')
    print('-------------------------')
    print(DataFrame.var(axis=0))
    
    #Etiqueta de componentes a calcular
    # ==========================================================================
    Etiquetas=[]
    for i in range(len(DataFrame.keys())):
        Etiquetas.append('PC'+str(i+1))
    
    # Entrenamiento modelo PCA con escalado de los datos
    # ==============================================================================
    pca_pipe = make_pipeline(StandardScaler(), PCA())
    pca_pipe.fit(DataFrame)
    
    # Se extrae el modelo entrenado del pipeline
    modelo_pca = pca_pipe.named_steps['pca']
    
    dfPca=pd.DataFrame(
        data    = modelo_pca.components_,
        columns = DataFrame.columns,
        index   = Etiquetas
    )
    
    print(dfPca)
    
    # Heatmap componentes
    # ==============================================================================
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
    componentes = modelo_pca.components_
    plt.imshow(componentes.T, cmap='viridis', aspect='auto')
    plt.yticks(range(len(DataFrame.columns)), DataFrame.columns)
    plt.xticks(range(len(DataFrame.columns)), np.arange(modelo_pca.n_components_) + 1)
    plt.grid(False)
    plt.colorbar();
    
    
    # Porcentaje de varianza explicada por cada componente
    # ==============================================================================
    print('----------------------------------------------------')
    print('Porcentaje de varianza explicada por cada componente')
    print('----------------------------------------------------')
    print(modelo_pca.explained_variance_ratio_)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.bar(
        x      = np.arange(modelo_pca.n_components_) + 1,
        height = modelo_pca.explained_variance_ratio_
    )
    
    for x, y in zip(np.arange(len(DataFrame.columns)) + 1, modelo_pca.explained_variance_ratio_):
        label = round(y, 2)
        ax.annotate(
            label,
            (x,y),
            textcoords="offset points",
            xytext=(0,10),
            ha='center'
        )
    
    ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
    ax.set_ylim(0, 1.1)
    ax.set_title('Porcentaje de varianza explicada por cada componente')
    ax.set_xlabel('Componente principal')
    ax.set_ylabel('Por. varianza explicada');
    
    # Porcentaje de varianza explicada acumulada
    # ==============================================================================
    prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
    print('------------------------------------------')
    print('Porcentaje de varianza explicada acumulada')
    print('------------------------------------------')
    print(prop_varianza_acum)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.plot(
        np.arange(len(DataFrame.columns)) + 1,
        prop_varianza_acum,
        marker = 'o'
    )
    
    for x, y in zip(np.arange(len(DataFrame.columns)) + 1, prop_varianza_acum):
        label = round(y, 2)
        ax.annotate(
            label,
            (x,y),
            textcoords="offset points",
            xytext=(0,10),
            ha='center'
        )
        
    ax.set_ylim(0, 1.1)
    ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
    ax.set_title('Porcentaje de varianza explicada acumulada')
    ax.set_xlabel('Componente principal')
    ax.set_ylabel('Por. varianza acumulada');
    
    proyecciones = pca_pipe.transform(X=DataFrame)
    proyecciones = pd.DataFrame(
        proyecciones,
        columns = Etiquetas,
        index   = DataFrame.index
    )
    print(proyecciones.head())
    
    return proyecciones,prop_varianza_acum

proyecciones,prop_varianza_acum=ACP(Per_Alpha.iloc[:,:-1])

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
x=np.array(proyecciones.loc[:,'PC1'])
y=np.array(proyecciones.loc[:,'PC2'])
z=np.array(proyecciones.loc[:,'PC3'])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
color=Dataframe.loc[:,'Cohort']
ax.scatter(x,y,z,c=color,alpha=.5)
plt.show()
