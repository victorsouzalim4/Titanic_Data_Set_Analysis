
# 🚢 Análise da Base de Dados do Titanic com Inteligência Artificial

Este projeto tem como objetivo aplicar e comparar diferentes técnicas de **Inteligência Artificial (IA)**, **Machine Learning** e **mineração de dados** sobre a clássica base do [Titanic (Kaggle)](https://www.kaggle.com/c/titanic), a fim de entender os fatores que mais influenciaram a sobrevivência dos passageiros.

Além de modelos supervisionados, também são exploradas abordagens não supervisionadas, regras de associação e uma **rede neural implementada do zero** via submódulo.

---

## 🧠 Algoritmos Aplicados

### 🔍 Classificação Supervisionada
- Random Forest
- Rede Neural (implementação manual com backpropagation)

### 📊 Agrupamento (Clustering)
- K-Means
- DBSCAN

### 📚 Regras de Associação
- Apriori (via `mlxtend`)

---

## 🧪 Objetivos

- Prever a variável `Survived` com diferentes algoritmos supervisionados.
- Identificar agrupamentos naturais de passageiros por perfis semelhantes.
- Extrair regras frequentes e associações interpretáveis com foco explicativo.
- Comparar desempenho, interpretabilidade, limitações e aplicabilidades de cada abordagem.

---

## 🧩 Estrutura do Projeto

```
Titanic_Data_Set_Analysis/
│
├── main.py                         ← Execução principal de todos os testes
|
├── Analysis/                       ← Análises e visualizações
│   ├── Apriori/
│   ├── DBSCAN/
│   ├── Kmeans/
|   ├── Neural_network/
|   ├── Random_forest/
│
├── Algorithms/                     ← Submódulo: Rede Neural manual
│   ├── Apriori/
│   ├── DBSCAN/
│   ├── Kmeans/
|   ├── Neural_network/
|   ├── Random_forest/
│
├── Utils/                          ← Utilitários comuns
│   ├── data_set_reader.py
│   ├── pre_processing.py
│   └── metrics_visual.py
│
├── Titanic_data_set/              ← Arquivos CSV de treino e teste
│
│
├── requirements.txt
└── README.md
```

---

## 🔧 Como Executar

### 📥 1. Clone o repositório com submódulo

```bash
git clone --recurse-submodules https://github.com/victorsouzalim4/Titanic_Data_Set_Analysis.git
cd Titanic_Data_Set_Analysis
```

> Esqueceu o `--recurse-submodules`? Use:
```bash
git submodule update --init --recursive
```

---

### 🧬 2. Crie e ative o ambiente virtual

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 📦 3. Instale as dependências

```bash
pip install -r requirements.txt
```

---

### ▶️ 4. Execute os testes

```bash
python main.py
```

As análises incluem:
- Métricas de classificação (acurácia, F1, ROC)
- Matrizes de confusão
- Visualizações t-SNE e silhueta
- Matriz de contingência entre clusters
- Regras de associação e grafo interativo

---

## 📂 Sobre o Submódulo `Algorithms/`

Este submódulo Git contém uma **rede neural totalmente implementada do zero**, com forward propagation, retropropagação e suporte a múltiplas camadas. Ele é utilizado neste projeto como benchmark interpretável e educativo.

> Submódulo vinculado ao repositório:  
> [victorsouzalim4/Neural_Networks_Sub](https://github.com/victorsouzalim4/Neural_Networks_Sub)

---

## 📊 Resultados e Conclusões

- **Random Forest** e **Rede Neural** obtiveram os melhores resultados classificatórios, com destaque para a rede em recall da classe minoritária (`Survived`).
- **K-Means** e **DBSCAN** geraram agrupamentos interpretáveis, mas com baixa correlação com a variável alvo.
- O algoritmo **Apriori** revelou regras fortes e explicáveis, como a forte associação entre `Sex_female` e `Survived_1`.

---

## 🧠 Lições Aprendidas

Este projeto reforça a importância de combinar diferentes estratégias de IA:

- Algoritmos supervisionados para previsão confiável.
- Agrupamento para análise exploratória e descoberta de padrões.
- Regras de associação para explicabilidade e descoberta de relações ocultas.

---

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues, sugerir melhorias ou enviar pull requests.

---

## 📜 Licença

Distribuído sob a licença [MIT](LICENSE).
