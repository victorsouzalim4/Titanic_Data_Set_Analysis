
# 🚢 Análise da Base de Dados do Titanic com Inteligência Artificial

Este projeto tem como objetivo estudar e aplicar diferentes técnicas de **Inteligência Artificial (IA)** e **Machine Learning** sobre a clássica base de dados do [Titanic (Kaggle)](https://www.kaggle.com/c/titanic), analisando os fatores que influenciam na sobrevivência dos passageiros.

Além de algoritmos clássicos, este repositório também explora uma implementação **manual de rede neural** como benchmark comparativo.

---

## 🧩 Técnicas de IA utilizadas

- Regressão Logística
- Árvores de Decisão (Decision Tree)
- Florestas Aleatórias (Random Forest)
- KNN
- Redes Neurais (Backpropagation manual, via submódulo)
- Análise exploratória de dados (EDA)
- Visualização de resultados

---

## 📁 Estrutura do Projeto

```
Titanic_Data_Set_Analysis/
│
├── main.py                         ← Execução principal dos testes
├── models/                         ← Algoritmos de IA implementados
│   ├── decision_tree.py
│   ├── logistic_regression.py
│   └── ...
├── analysis/                       ← Códigos de exploração e visualização
│   └── ...
├── data/                           ← Dados CSVs (se aplicável)
│
├── Algorithms/                     ← Submódulo: Rede Neural manual
│   ├── forward.py
│   ├── neuron.py
│   └── ...
├── requirements.txt
└── README.md
```

---

## 🧠 Submódulo `Algorithms/`: Rede Neural Manual

Este projeto utiliza um submódulo Git chamado `Algorithms/`, que contém uma implementação didática de uma **rede neural com backpropagation feita do zero** em Python. Essa rede é utilizada como comparativo com os modelos tradicionais.

> O submódulo está vinculado ao repositório:  
> [victorsouzalim4/Neural_Networks_Sub](https://github.com/victorsouzalim4/Neural_Networks_Sub)

---

## 🧪 Como rodar o projeto

### 📥 1. Clone o projeto com submódulos

```bash
git clone --recurse-submodules https://github.com/seu-usuario/Titanic_Data_Set_Analysis.git
cd Titanic_Data_Set_Analysis
```

> Esqueceu o `--recurse-submodules`? Execute:
```bash
git submodule update --init --recursive
```

---

### 🧬 2. Crie e ative o ambiente virtual

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS:
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

Você verá as métricas de desempenho, erros médios, classificações e comparações entre os modelos testados, incluindo a rede neural.

---

## 🛠 Sobre o submódulo

Se você fizer alterações dentro de `Algorithms/`, lembre-se de:

1. Committar dentro do submódulo:
```bash
cd Algorithms
git add .
git commit -m "Modificação na rede neural"
git push origin main
```

2. Atualizar o commit referenciado no repositório principal:
```bash
cd ..
git add Algorithms
git commit -m "Atualiza commit do submódulo"
git push
```

---

## 🤝 Contribuições

Sugestões, issues e pull requests são bem-vindos. Este projeto é acadêmico e exploratório, com o objetivo de aprendizado prático em IA.

---

## 📜 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
