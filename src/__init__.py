"""dynkin_search -- Objetivo 1 (MM845): RL para diagramas de Dynkin/Coxeter.

Módulos desta entrega:
    coxeter.py     -- representação de diagramas, matriz de Cartan, teste
                       de esfericidade, utilidades de grafo (union-find,
                       componentes conexas, aciclicidade), e utilidades de
                       simetria por permutação (código canônico de
                       isomorfismo, contagem de automorfismos, tamanho de
                       órbita sob S_n).
    exhaustive.py  -- teste exaustivo GENUÍNO da condição esférica (N <= 6):
                       todo diagrama, cíclico ou não, é de fato testado --
                       o fato "todo diagrama esférico conexo é uma árvore"
                       não é assumido em nenhum momento, apenas checado a
                       posteriori (cross-check contra o método antigo,
                       baseado em florestas, mantido só para validação).
                       O que É usado é invariância por permutação: diagramas
                       relacionados por uma relabeling dos vértices são
                       esféricos juntos ou não, então basta testar um
                       representante por classe de isomorfismo e ponderar
                       pelo tamanho da órbita. As classes são geradas por
                       construção incremental vértice-a-vértice (geração
                       isomorph-free), vetorizada em lotes para viabilizar
                       N = 6 (~1,6 milhão de classes, ~1,07 bilhão de
                       diagramas rotulados, ~9.826 esféricos -- validado
                       contra o método antigo).
    datagen.py     -- amostragem i.i.d. dos diagramas iniciais (Dataset A
                       uniforme / Dataset B esparso) até N = 8, e as
                       estatísticas estruturais pré-treino.

Próximas entregas: ambiente de RL (env.py), política GNN (policy.py),
treino PPO (train_ppo.py) e identificação de tipos + visualização
(analyze.py).
"""
