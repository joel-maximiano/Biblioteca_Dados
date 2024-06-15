import functools

import pandas as pd
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.decomposition import PCA


class DataFrame:
    def __init__(self, dados=None):
        if dados is None:
            self.dados = pd.DataFrame()  # Criar um DataFrame vazio se nenhum dado for fornecido
        elif isinstance(dados, dict):
            self.dados = pd.DataFrame.from_dict(dados)  # Criar DataFrame a partir de um dicionário
        else:
            raise TypeError("Os dados fornecidos devem ser um dicionário.")

        self.colunas = list(self.dados.columns) if not self.dados.empty else []

    def __str__(self):
        return self.dados.__str__()

    def unir(self, coluna1, coluna2, nome_nova_coluna):
        """
        Une duas colunas (ou séries) existentes no DataFrame em uma nova coluna.

        Args:
        - coluna1 (str): Nome da primeira coluna.
        - coluna2 (str): Nome da segunda coluna.
        - nome_nova_coluna (str): Nome para a nova coluna que será criada.
        """
        if coluna1 not in self.colunas or coluna2 not in self.colunas:
            raise KeyError("As colunas fornecidas não existem no DataFrame.")

        self.dados[nome_nova_coluna] = self.dados[coluna1] + self.dados[coluna2]
        self.colunas.append(nome_nova_coluna)

    def __repr__(self):
        return f"DataFrame: linhas={len(self.dados)}, colunas={len(self.colunas)}"

    def cabeca(self, n=5):
        return self.dados.head(n)

    def cauda(self, n=5):
        return self.dados.tail(n)

    def info(self):
        return self.dados.info()

    def corpo(self):
        return self.dados.shape

    def colunas(self):
        return self.dados.columns.tolist()

    def converter_para_maiusculas(self, coluna):
        # Converte os valores de uma coluna para maiúsculas
        if coluna not in self.colunas:
            raise KeyError(f"Coluna '{coluna}' não encontrada no DataFrame.")
        self.dados[coluna] = self.dados[coluna].str.upper()
        return self

    def converter_para_minusculas(self, coluna):
        # Converte os valores de uma coluna para minúsculas
        if coluna not in self.colunas:
            raise KeyError(f"Coluna '{coluna}' não encontrada no DataFrame.")
        self.dados[coluna] = self.dados[coluna].str.lower()
        return self

    def aplicar_funcao(self, funcao, eixo=0):
        # Aplica uma função ao longo do DataFrame
        return self.dados.apply(funcao, axis=eixo)



    def calcular_correlacao(self, metodo='pearson'):
        # Calcula a correlação entre colunas numéricas do DataFrame
        return self.dados.corr(method=metodo)

    def plotar(self, x=None, y=None, tipo='linha', **kwargs):
        # Função para plotar dados usando matplotlib ou seaborn
        if tipo == 'linha':
            if x is None or y is None:
                raise ValueError("Especifique colunas para x e y.")
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=self.dados, x=x, y=y, **kwargs)
            plt.title(f"Gráfico de Linha de {y} por {x}")
            plt.show()
        elif tipo == 'histograma':
            plt.figure(figsize=(10, 6))
            self.dados.hist(**kwargs)
            plt.title("Histograma")
            plt.show()
        else:
            raise ValueError(f"Tipo de plot '{tipo}' não suportado.")

    def exportar_csv(self, caminho):
        # Exporta o DataFrame para um arquivo CSV
        self.dados.to_csv(caminho, index=False)
        print(f"DataFrame exportado para {caminho}")

    def importar_csv(self, caminho):
        # Importa um arquivo CSV para um DataFrame
        dados_importados = pd.read_csv(caminho)
        return DataFrame(dados_importados, colunas=dados_importados.columns)

    def aplicar_funcao_paralela(self, funcao, eixo=0):
        # Aplica uma função ao longo do DataFrame usando processamento paralelo
        num_cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(num_cores) as pool:
            resultados = pool.apply(funcao, (self.dados,))
        return resultados

    def calcular_correlacao_numpy(self, metodo='pearson'):
        # Calcula a correlação entre colunas numéricas do DataFrame usando NumPy
        matriz_dados = np.array(self.dados)
        correlacao = np.corrcoef(matriz_dados, rowvar=False, method=metodo)
        return correlacao

    def analise_estatistica_avancada(self):
        # Calcula estatísticas descritivas detalhadas para colunas numéricas
        estatisticas = self.dados.describe(include='all')
        return estatisticas

    def aplicar_modelo(self, modelo):
        # Aplica um modelo treinado ao DataFrame
        resultados = modelo.predict(self.dados)
        return resultados

    def exportar_excel(self, caminho):
        # Exporta o DataFrame para um arquivo Excel
        self.dados.to_excel(caminho, index=False)
        print(f"DataFrame exportado para {caminho}")

    def importar_excel(self, caminho):
        # Importa um arquivo Excel para um DataFrame
        dados_importados = pd.read_excel(caminho)
        return DataFrame(dados_importados, colunas=dados_importados.columns)

    def amostrar_dados(self, frac=0.5, substituicao=False, random_state=None):
        # Amostra aleatória de dados do DataFrame
        dados_amostrados = self.dados.sample(frac=frac, replace=substituicao, random_state=random_state)
        return DataFrame(dados_amostrados, colunas=self.colunas, indice=self.indice)

    def aplicar_transformacao(self, transformacao):
        # Aplica uma transformação específica aos dados
        dados_transformados = transformacao.transform(self.dados)
        return DataFrame(pd.DataFrame(dados_transformados, columns=self.dados.columns), colunas=self.colunas,
                         indice=self.indice)

    def validar_dados(self):
        # Validação básica de integridade dos dados
        dados_validos = self.dados.dropna()
        return DataFrame(dados_validos, colunas=self.colunas, indice=self.indice)

    def processar_imagens(self, coluna_imagens):
        # Processamento básico de imagens (exemplo: redimensionamento)
        pass  # Implementação fictícia para este exemplo

    def detectar_anomalias(self, metodo='gaussian'):
        # Detecção de anomalias nos dados
        if metodo == 'gaussian':
            # Implementar método de detecção de anomalias baseado em distribuição gaussiana
            pass  # Implementação fictícia para este exemplo
        else:
            raise ValueError("Método de detecção de anomalias não suportado.")

        # Retornar um DataFrame vazio como exemplo
        return DataFrame(pd.DataFrame(columns=self.dados.columns), colunas=self.colunas, indice=self.indice)

    def vetorizar_texto(self, coluna_texto):
        # Vetorização de texto usando TF-IDF
        tfidf_vectorizer = TfidfVectorizer()
        matriz_vetorizada = tfidf_vectorizer.fit_transform(self.dados[coluna_texto])
        colunas_vetorizadas = tfidf_vectorizer.get_feature_names_out()
        return DataFrame(pd.DataFrame(matriz_vetorizada.toarray(), columns=colunas_vetorizadas), colunas=self.colunas,
                         indice=self.indice)
    def reduzir_dimensionalidade(self, metodo='pca', n_componentes=None):
        # Redução de dimensionalidade dos dados
        if metodo == 'pca':
            pca = PCA(n_components=n_componentes)
            dados_reduzidos = pca.fit_transform(self.dados)
            return DataFrame(pd.DataFrame(dados_reduzidos, columns=[f'componente_{i+1}' for i in range(n_componentes)]), colunas=self.colunas, indice=self.indice)
        else:
            raise ValueError("Método de redução de dimensionalidade não suportado.")

    def plotar_componentes_principais(self, coluna_x, coluna_y):
        # Plotagem dos componentes principais após PCA
        plt.figure(figsize=(10, 6))
        plt.scatter(self.dados[coluna_x], self.dados[coluna_y])
        plt.title('Plotagem de Componentes Principais')
        plt.xlabel(coluna_x)
        plt.ylabel(coluna_y)
        plt.show()

    def aplicar_funcao_elemento(self, funcao):
        # Aplica uma função a cada elemento do DataFrame
        dados_aplicados = self.dados.applymap(funcao)
        return DataFrame(dados_aplicados, colunas=self.colunas, indice=self.indice)

    def adicionar_coluna(self, nome_coluna, valores):
        # Adiciona uma nova coluna ao DataFrame
        if len(valores) != len(self.dados):
            raise ValueError("O número de valores não corresponde ao número de linhas no DataFrame.")
        self.dados[nome_coluna] = valores
        self.colunas.append(nome_coluna)
        return self

    def remover_coluna(self, nome_coluna):
        # Remove uma coluna do DataFrame
        if nome_coluna not in self.colunas:
            raise KeyError(f"Coluna '{nome_coluna}' não encontrada no DataFrame.")
        self.dados = self.dados.drop(columns=nome_coluna)
        self.colunas.remove(nome_coluna)
        return self

    @functools.lru_cache(maxsize=None)
    def operacao_pesada(self, argumento):
        # Exemplo de operação pesada que pode ser cacheada
        resultado = ...  # Operação pesada baseada no argumento
        return resultado