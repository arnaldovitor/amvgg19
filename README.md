# Appearance and Motion VGG-19
Método de detecção de eventos anômalos em vídeos inspirado no artigo Detecting anomalous events in videos by learning deep representations of appearance and motion.

A. process_videos.py
> Módulo de extração de frames. Recebe vídeos e salva os frames dos mesmos nas representações RGB e Optical Flow.

B. feature_extractor.py
> Módulo de extração de característica. Recebe os frames nas representações RGB e Optical Flow e processa os mesmos usando uma VGG-19 pré-treinada no conjunto ImageNet.

C. vbow.py
> Módulo de Visual Bag of Words. Recebe os feature vectors obtidos pela VGG-19 e retorna uma lista de histogramas afim de serem utilizados posteriormente em um classificador.

D. pca.py
> Módulo de redução de dimensionalidade afim de fazer alguns testes de melhora de accuracy, usa o PCA da biblioteca scikit-learn. 

E. classifiers.py
> Módulo com classificadores e funções que ajudam a avaliar a eficácia do método proposto em comparação a outros.

F. util.py
> Módulo com funções úteis para todos os outros módulos do método.

G. youtube.py
> Módulo com função para baixar vídeos do YouTube caso deseje testar o método em outras fontes.

H. view.py
> Módulo para visualizar os vídeos de teste após a classificação.
