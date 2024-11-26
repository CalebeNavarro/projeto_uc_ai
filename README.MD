# **Análise do Desempenho Estudantil**

## **Descrição do Projeto**
O objetivo desta pesquisa é estudar a base para entende-la em certas características socioeconômicas e educacionais através da análise do conjunto de dados Desempenho dos Estudantes. Três modelos foram usados para prever os níveis de desempenho.
1. **Árvores de Decisão**
2. **Naive Bayes**
3. **Redes Neurais**

Os objetivos do projeto de pesquisa são, determinar qual modelo apresenta o melhor desempenho na previsão da categoria de desempenho de um estudante a partir de três categorias possíveis, sendo **abaixo** da média, **média** e **acima* da média.


## **Descrição do Dataset**
O dataset contém informações sobre alunos, como:
- Características Socioeconômicas:
  - Gênero (`gender`)
  - Grupo etnico (`race_ethnicity`)
  - Nível de educação dos país (`parental_level_of_education`)
  - Tipo de almoço (`lunch`)
  - Participação em curso de preparação para provas (`test_preparation_course`)
- Desempenho Acade^mico:
  - Notas em matemática (`math_score`), leitura (`reading_score`) e escrita (`writing_score`)
  - Nota total (`total_score`) e media (`average_score`).

A variável **target** definida para este proejto foi a categoria de desempenho, criada a partir da média das notas:
- Low: Média < 60
- Medium: 60 =< Média < 80
- High: Média => 80

## **Etapas do Projeto**

### 1. **Preparação dos Dados**
- Criação da variável categórica `performance_category` baseada na média das notas.
- Codificação de variáveis categóricas usando Label Encoding e One-Hot Encoding.
- Divisão do dataset em 80% treino e 20% teste.

### 2. **Modelos de IA Aplicados**
Escolhi as três primeira opções, sendo elas:

- Árvores de Decisão:
- Naive Bayes:
- Redes Neurais.

### 3. **Avaliação dos Modelos**
O foco da avaliaçõa foi no target performance_category como já vimos acima, porém para ter algo mais completo os modelos também foram avaliados com as métricas:
- Precisão: Percentual de predições corretas.
- F1-Score: Média harmônica entre precisão e recall.
- Matriz de Confusão: Visualiza os erros por classe.


## **Resultados**
Resultado fixo na análise do arquivo Cleaned_Students_Performance.csv. Se encontra dentro do projeto.

| Modelo           | Precisão | F1-Score | Observações                                                                 |
|-------------------|----------|----------------------------|-----------------------------------------------------------------------------|
| Árvores de Decisão | 93%      | 93%                        | Excelente desempenho geral, com ótima recuperação e precisão no balanceamento.             |
| Naive Bayes       | 94%      | 94%                        | Melhor precisão e menos erros, particularmente nas categorias "Low" e "High". |
| Redes Neurais     | 81%      | 80%                        | Desempenho irregular na categoria "High", porém aceitável nas demais.

---

## **Conclusão**
A escolha do modelo Naive Bayes para este projeto foi baseada em:
1. Mostrar maior precisão geral (94%).
2. Ter um bom rendimento em todas as disciplinas, com um número reduzido de falhas como modelo acima.
3. Este tipo de análise requer simplicidade e eficiência.

## **Como Executar o Código**
### 1. Pré-requisitos
python

### 2. Passos
1. Clone este repositório:
   ```bash
   git clone git@github.com:CalebeNavarro/projeto_uc_ai.git
    ```
    
2. Faça a instalação:
  ```bash
  pip install -r requirement.txt
  ```

3. Rode o programa
```bash
python .\main.py
```
