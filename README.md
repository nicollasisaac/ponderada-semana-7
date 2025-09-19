
# Ponderada Semana 7 — IEEE-CIS Fraud Detection com LSTM

Trabalho desenvolvido por Henrique Cox e Nicollas Isaac. Nosso objetivo foi montar um pipeline completo para detecção de fraudes no dataset IEEE-CIS, desde ingestão e tratamento de dados até modelagem com LSTM e avaliação orientada a métricas adequadas para classe rara.

## Visão geral

- Dataset altamente desbalanceado (teste: 96,5% não-fraude vs 3,5% fraude).
- Pipeline de pré-processamento com imputação, padronização e validações gráficas.
- Modelo LSTM simples como linha de base, treinado em 5 épocas.
- Avaliação com ROC, PR, varredura de limiar e análise de trade-offs.
- Recomendação inicial de threshold operacional com base nos resultados.

## Equipe

- Henrique Cox  
- Nicollas Isaac

## Como rodar

1. Python 3.10+ e virtualenv recomendados
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt

2. Abra o notebook em `notebooks/` e execute as células na ordem.

## Dados e ingestão

* Carregamos os CSVs a partir de links do Google Drive sem depender de caminho local.
* Implementamos um loader robusto:

  * aceita link completo ou ID do Drive
  * tenta leitura direta via URL
  * fallback com `gdown` para lidar com arquivos grandes e token de confirmação
  * leitura com `compression='infer'` e `low_memory=False`
  * diagnóstico caso o Drive retorne HTML em vez de CSV

## Exploração e tratamento de dados

* Checagem de valores ausentes em todas as colunas de `train_identity` e `train_transaction`.
* Remoção de colunas com mais de 90% de valores nulos para reduzir ruído e sparsity.
* Imputação:

  * numéricas por mediana (robusta a outliers)
  * categóricas por moda
* Garantimos `isFraud` como numérica e removemos linhas com alvo ausente.
* Distribuição de `TransactionAmt` antes e depois da imputação para checar impacto no shape da variável.

## Preparação para modelagem

* Selecionamos apenas features numéricas para a primeira linha de base.
* Substituímos ±inf por NaN e reimputamos se necessário.
* Padronização com `StandardScaler` (média 0 e desvio 1).
* Reorganizamos os dados para LSTM com sequência de 1 timestep: `(amostras, 1, n_features)`.
* Split estratificado em treino e teste (80/20) preservando a proporção de fraudes.

## Modelo

* Arquitetura LSTM:

  * `LSTM(64, return_sequences=True)` → `Dropout(0.2)` → `LSTM(32)` → `Dropout(0.2)` → `Dense(1, activation='sigmoid')`
* Compilação:

  * `loss = binary_crossentropy`
  * `optimizer = Adam`
  * métricas de treino: `accuracy` (complementadas por métricas focadas na classe positiva na avaliação)
* Treino:

  * `epochs = 5` como ponto de partida
  * `batch_size = 512`
* Curvas de aprendizado mostraram queda estável de loss em treino e validação, sem sinais fortes de overfitting nesse horizonte curto.

## Avaliação

* Distribuição da classe no teste:

  * 0 (não-fraude): 113.975 amostras (96,50%)
  * 1 (fraude): 4.133 amostras (3,50%)
* Threshold padrão 0,50

  * Matriz: TP=1.634, FP=245, FN=2.499, TN=113.730
  * Classe 1: precision=0,8696, recall=0,3954, F1=0,5436
  * Leitura: poucos falsos positivos, mas muitas fraudes passam sem alerta
* Threshold ótimo por F1 ≈ 0,25

  * Matriz: TP=1.974, FP=644, FN=2.159, TN=113.331
  * Classe 1: precision=0,7540, recall=0,4776, F1=0,5848
  * Acurácia geral: 0,9763
  * Leitura: equilíbrio melhor entre pegar fraudes e manter precisão operacional
* Threshold ótimo por Youden J ≈ 0,04

  * Matriz: TP=3.069, FP=13.246, FN=1.064, TN=100.729
  * Classe 1: precision=0,1881, recall=0,7426
  * Leitura: cobertura alta de fraudes com custo elevado em falsos positivos
* Curvas

  * ROC e AUC-ROC para separabilidade global
  * Precisão-Recall e Average Precision (PR AUC), mais informativas para classe rara
  * Varredura de threshold com métricas vs. cutoff para guiar a operação
  * Curva de calibração para checar aderência das probabilidades

## O que aprendemos

* A acurácia por si só não reflete bem a performance em fraude pela base ser muito desbalanceada.
* O threshold muda a operação:

  * 0,50 é conservador, prioriza precisão e reduz custo de revisão, mas deixa passar fraudes
  * 0,25 melhora o F1 da classe positiva e oferece um compromisso mais saudável
  * 0,04 quase não deixa passar fraude, porém “inunda” a fila com falsos positivos
* A LSTM com 1 timestep se comporta como uma rede sobre vetores estáticos; para extrair o diferencial da recorrência precisamos de janelas temporais por usuário ou dispositivo, com dependências entre eventos.

## Hipóteses e próximos passos

* Otimizar o threshold por custo esperado de FP e FN, não só por F1, para alinhar com o negócio.
* Calibrar probabilidades (Platt/Isotonic) e reavaliar thresholds após calibração.
* Analisar métricas por segmento (canal, device, país, valor de transação) e definir thresholds específicos por segmento quando fizer sentido.
* Construir sequências temporais reais por entidade para explorar padrões de recorrência, tempos entre eventos e sazonalidade.
* Comparar com baselines tabulares fortes (XGBoost/LightGBM) para medir o ganho real do modelo sequencial.
* Considerar `class_weight`, focal loss ou reamostragem para aumentar recall sem deteriorar demais a precisão.

## Recomendação atual de operação

Sem função de custo explícita, recomendamos iniciar com threshold ≈ 0,25 por entregar melhor F1 na classe positiva e um trade-off equilibrado entre cobertura de fraude e volume de alertas. Ajustes finos devem ser guiados por custo real de revisão e pelas métricas por segmento.
