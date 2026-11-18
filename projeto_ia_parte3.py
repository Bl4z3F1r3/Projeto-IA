# %% [markdown]
# # Projeto de IA – Parte 3: Classificação de Filmes de Animação
# **FATEC Rodrigues de Abreu – Bauru/SP**
# Autores: João Victor Godoy Bortolotti e Raul Pavan
# Orientadora: Profa. Dra. Patricia Bellin Ribeiro
#
# Objetivo: classificar filmes de animação em **Bom (1, nota TMDB ≥ 7)** ou **Ruim (0, nota < 7)**
# a partir de atributos numéricos, categóricos e textuais da base tratada na Parte 2.
#
# Algoritmos: **Multi-Layer Perceptron (MLP)** e **Gradient Boosting** (Scikit-Learn),
# comparados a um baseline (classe majoritária).

# %%
import json, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, RocCurveDisplay)
warnings.filterwarnings('ignore')
SEED = 42
os.makedirs('figs', exist_ok=True)
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10})

# %% [markdown]
# ## 1. Carregamento da base tratada (Parte 2)

# %%
df = pd.read_csv('CSV_Tratado_Projeto_IA_Parte2.csv')
print(df.shape)
print(df['Classe'].value_counts().rename({0: 'Ruim (0)', 1: 'Bom (1)'}))
print((df['Classe'].value_counts(normalize=True) * 100).round(2))

# %% [markdown]
# ## 2. Engenharia de atributos
#
# Para evitar **vazamento de dados (data leakage)** foram descartadas colunas derivadas da nota
# (`TMDB_Rating`, que define a classe, e `Hidden_Gem`, calculada a partir dela), além de
# identificadores, URLs e colunas com mais de 85% de valores ausentes (orçamento, bilheteria, ROI).
# Colunas de alta cardinalidade (estúdio, diretor, elenco) não foram usadas nesta fase.

# %%
df = df.copy()
df['Genre_txt'] = df['Genre'].fillna('').str.replace('|', ',', regex=False)
df['Overview'] = df['Overview'].fillna('')
df['Log_Vote_Count'] = np.log1p(df['TMDB_Vote_Count'])
df['Log_Popularity'] = np.log1p(df['TMDB_Popularity'])
df['Has_Collection'] = df['Belongs_To_Collection'].notna().astype(int)
df['Has_Budget'] = df['Budget_Million_USD'].notna().astype(int)
df['Has_Box_Office'] = df['Box_Office_Million_USD'].notna().astype(int)
df['Remake'] = (df['Live_Action_Remake'] == 'Yes').astype(int)
df['Target_Audience'] = df['Target_Audience'].str.replace('|', '/', regex=False)
top_lang = df['Original_Language'].value_counts().index[:15]
df['Lang'] = df['Original_Language'].where(df['Original_Language'].isin(top_lang), 'other')
mpaa_top = df['MPAA_Rating'].value_counts().index[:10]
df['MPAA'] = df['MPAA_Rating'].where(df['MPAA_Rating'].isin(mpaa_top), 'other')
for c in ['Is_TV_Compilation', 'Is_Adult_Content', 'Is_Future_Release']:
    df[c] = df[c].astype(int)

NUM = ['Release_Year', 'Movie_Length_Minutes', 'Log_Vote_Count', 'Log_Popularity',
       'Has_Collection', 'Has_Budget', 'Has_Box_Office', 'Remake',
       'Is_TV_Compilation', 'Is_Adult_Content', 'Is_Future_Release']
CAT = ['Animation_Style', 'Target_Audience', 'Lang', 'MPAA', 'Era', 'Popularity_Tier']
X = df[NUM + CAT + ['Genre_txt', 'Overview']]
y = df['Classe']
print(X.shape)

# %% [markdown]
# ## 3. Divisão dos dados
#
# Divisão **estratificada 80/20** (treino/teste) + **validação cruzada estratificada de 5 folds** no treino.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)
split_info = {'treino': len(X_train), 'teste': len(X_test),
              'bom_treino': float(y_train.mean()), 'bom_teste': float(y_test.mean())}
print(split_info)

# %% [markdown]
# ## 4. Pré-processamento em pipeline (sem vazamento entre treino e teste)
# A saída é densa (`sparse_threshold=0`), exigida pelo Gradient Boosting baseado em histogramas.

# %%
def make_preprocessor():
    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                         ('ohe', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([
        ('num', num_pipe, NUM),
        ('cat', cat_pipe, CAT),
        ('genre', TfidfVectorizer(tokenizer=lambda s: [t.strip().lower() for t in s.split(',') if t.strip()],
                                  token_pattern=None, lowercase=False), 'Genre_txt'),
        ('ovw', TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2),
                                min_df=3, sublinear_tf=True), 'Overview'),
    ], sparse_threshold=0)

models = {
    'Baseline (classe majoritária)': DummyClassifier(strategy='most_frequent'),
    'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', alpha=1e-2,
                         learning_rate_init=1e-3, early_stopping=True, validation_fraction=0.1,
                         n_iter_no_change=8, max_iter=200, random_state=SEED),
    'Gradient Boosting': HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=6,
                                                        l2_regularization=1.0, class_weight='balanced',
                                                        early_stopping=True, validation_fraction=0.1,
                                                        n_iter_no_change=20, random_state=SEED),
}
pipes = {n: Pipeline([('prep', make_preprocessor()), ('clf', m)]) for n, m in models.items()}

# %% [markdown]
# ## 5. Validação cruzada (5 folds) no conjunto de treino

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_results = {}
for name, pipe in pipes.items():
    t0 = time.time()
    r = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    cv_results[name] = {m: (float(r['test_' + m].mean()), float(r['test_' + m].std())) for m in scoring}
    print(name, f'{time.time() - t0:.0f}s')
cv_df = pd.DataFrame({n: {m: f'{v[0]:.3f} ± {v[1]:.3f}' for m, v in d.items()} for n, d in cv_results.items()}).T
cv_df

# %% [markdown]
# ## 6. Treinamento final e avaliação no conjunto de teste

# %%
test_results, preds, probas = {}, {}, {}
for name, pipe in pipes.items():
    pipe.fit(X_train, y_train)
    p = pipe.predict(X_test)
    pr = pipe.predict_proba(X_test)[:, 1]
    preds[name], probas[name] = p, pr
    test_results[name] = {
        'accuracy': accuracy_score(y_test, p),
        'precision': precision_score(y_test, p, zero_division=0),
        'recall': recall_score(y_test, p),
        'f1': f1_score(y_test, p),
        'roc_auc': roc_auc_score(y_test, pr),
        'cm': confusion_matrix(y_test, p).tolist(),
    }
    print(f'=== {name} ===')
    print(classification_report(y_test, p, target_names=['Ruim (0)', 'Bom (1)'], digits=3, zero_division=0))
pd.DataFrame(test_results).T.drop(columns='cm').round(3)

# %% [markdown]
# ## 7. Gráficos
# Paleta: cinza (baseline), azul-petróleo (MLP) e laranja (Gradient Boosting); hachuras garantem
# legibilidade também em impressão preto e branco.

# %%
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
CORES = {'Baseline (classe majoritária)': '#8c8c8c', 'MLP': '#1b6f7a', 'Gradient Boosting': '#e08a1e'}
HACH = {'Baseline (classe majoritária)': '', 'MLP': '//', 'Gradient Boosting': 'xx'}
plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.25, 'axes.axisbelow': True, 'font.family': 'DejaVu Sans'})
def limpa(ax):
    ax.spines[['top', 'right']].set_visible(False)

# 7.1 Distribuição da classe (barras horizontais)
fig, ax = plt.subplots(figsize=(6.4, 2.6))
vc = y.value_counts().sort_index()
ax.barh(['Bom (1)', 'Ruim (0)'], [vc[1], vc[0]], color=['#e08a1e', '#1b6f7a'], height=0.55)
for i, v in enumerate([vc[1], vc[0]]):
    ax.text(v + 150, i, f'{v:,} filmes ({v / len(y):.1%})'.replace(',', '.'), va='center')
ax.set_xlim(0, vc.max() * 1.45); ax.set_xlabel('Número de filmes'); ax.grid(axis='y', alpha=0); limpa(ax)
plt.tight_layout(); plt.savefig('figs/distribuicao_classe.png', dpi=200); plt.close()

# 7.2 Matrizes de confusão (contagem e % da classe real)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
for ax, name in zip(axes, ['MLP', 'Gradient Boosting']):
    cm = np.array(test_results[name]['cm']); pct = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(pct, cmap='YlGnBu', vmin=0, vmax=1); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]}\n({pct[i, j]:.1%})', ha='center', va='center', fontsize=11,
                    color='white' if pct[i, j] > 0.55 else 'black')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Ruim (0)', 'Bom (1)']); ax.set_yticklabels(['Ruim (0)', 'Bom (1)'])
    ax.set_xlabel('Classe prevista'); ax.set_ylabel('Classe real'); ax.set_title(name, fontweight='bold')
plt.tight_layout(); plt.savefig('figs/matrizes_confusao.png', dpi=200); plt.close()

# 7.3 Comparativo de métricas (barras horizontais agrupadas)
names = list(CORES)
keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
labels = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'ROC-AUC']
fig, ax = plt.subplots(figsize=(8, 5))
h = 0.26; yy = np.arange(len(keys))[::-1]
for i, n in enumerate(names):
    vals = [test_results[n][k] for k in keys]
    b = ax.barh(yy + (1 - i) * h, vals, h, color=CORES[n], hatch=HACH[n], edgecolor='white',
                label=n.replace(' (classe majoritária)', ''))
    ax.bar_label(b, labels=[f'{v:.3f}'.replace('.', ',') for v in vals], padding=3, fontsize=8)
ax.set_yticks(yy); ax.set_yticklabels(labels); ax.set_xlim(0, 1.0); ax.set_xlabel('Valor da métrica')
ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.12)); ax.grid(axis='y', alpha=0); limpa(ax)
plt.tight_layout(); plt.savefig('figs/comparativo_metricas.png', dpi=200); plt.close()

# 7.4 Curvas ROC e Precisão-Recall (lado a lado)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.1))
ap = {}
for n in names[1:]:
    fpr, tpr, _ = roc_curve(y_test, probas[n])
    a1.plot(fpr, tpr, color=CORES[n], lw=2, label=f"{n} (AUC = {test_results[n]['roc_auc']:.3f})".replace('.', ','))
    pr_, rc_, _ = precision_recall_curve(y_test, probas[n])
    ap[n] = float(average_precision_score(y_test, probas[n]))
    a2.plot(rc_, pr_, color=CORES[n], lw=2, label=f"{n} (AP = {ap[n]:.3f})".replace('.', ','))
a1.plot([0, 1], [0, 1], ':', color='#555', label='Aleatório (AUC = 0,500)')
a2.axhline(y_test.mean(), ls=':', color='#555', label=f'Aleatório (AP = {y_test.mean():.3f})'.replace('.', ','))
a1.set_xlabel('Taxa de falsos positivos'); a1.set_ylabel('Taxa de verdadeiros positivos'); a1.set_title('Curva ROC', fontweight='bold')
a2.set_xlabel('Recall'); a2.set_ylabel('Precisão'); a2.set_title('Curva Precisão-Recall', fontweight='bold')
for a in (a1, a2): a.legend(frameon=False, fontsize=8, loc='lower right' if a is a1 else 'upper right'); limpa(a)
plt.tight_layout(); plt.savefig('figs/curvas_roc_pr.png', dpi=200); plt.close()

# 7.5 Fluxogramas (horizontais, em etapas numeradas)
def fluxo(etapas, fn, ncol):
    nl = (len(etapas) + ncol - 1) // ncol
    fig, ax = plt.subplots(figsize=(9, 1.55 * nl)); ax.axis('off'); ax.set_xlim(0, ncol * 3); ax.set_ylim(0, nl * 2)
    for k, (t, sub) in enumerate(etapas):
        r, c = divmod(k, ncol); x, yb = c * 3 + 0.12, (nl - 1 - r) * 2 + 0.2
        ax.add_patch(FancyBboxPatch((x, yb), 2.4, 1.35, boxstyle='round,pad=0.02,rounding_size=0.1', fc='#eaf3f4', ec='#1b6f7a', lw=1.3))
        ax.text(x + 0.18, yb + 1.12, str(k + 1), fontsize=9, fontweight='bold', color='white', ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.25', fc='#1b6f7a', ec='none'))
        ax.text(x + 1.25, yb + 0.85, t, ha='center', va='center', fontsize=7.8, fontweight='bold', wrap=True)
        ax.text(x + 1.2, yb + 0.35, sub, ha='center', va='center', fontsize=6.6, color='#333')
        if c < ncol - 1 and k < len(etapas) - 1:
            ax.annotate('', xy=(x + 3, yb + 0.67), xytext=(x + 2.42, yb + 0.67), arrowprops=dict(arrowstyle='-|>', color='#1b6f7a', lw=1.3))
    plt.tight_layout(); plt.savefig(fn, dpi=200); plt.close()
fluxo([('Base original', '25.390 × 44 (Kaggle)'), ('Remoção sem nota', '9.272 registros'),
       ('Criação da Classe', 'Bom ≥ 7 | Ruim < 7'), ('Base tratada', '16.118 × 45')], 'figs/fluxo_preproc.png', 4)
fluxo([('Base tratada', '16.118 filmes'), ('Engenharia de atributos', 'sem vazamento'), ('Divisão 80/20', 'estratificada'),
       ('Pipeline', 'imputação, escala, TF-IDF'), ('Validação cruzada', '5 folds no treino'), ('Treino final', 'MLP | GB | baseline'),
       ('Avaliação no teste', 'métricas e matrizes')], 'figs/fluxo_metodo.png', 4)

# %% [markdown]
# ## 8. Importância dos atributos (permutação)
# Nem o MLP nem o HistGradientBoosting expõem importâncias nativas. Usa-se a **importância por
# permutação** sobre as colunas originais do conjunto de teste: embaralha-se cada coluna e mede-se a
# queda da ROC-AUC (quanto maior a queda, mais importante o atributo).

# %%
feat_cols = NUM + CAT + ['Genre_txt', 'Overview']
nice = {'Genre_txt': 'Genre (texto)', 'Overview': 'Overview (sinopse)'}
perm = {}
for name in ['MLP', 'Gradient Boosting']:
    r = permutation_importance(pipes[name], X_test, y_test, scoring='roc_auc',
                               n_repeats=5, random_state=SEED, n_jobs=1)
    perm[name] = pd.Series(r.importances_mean, index=[nice.get(c, c) for c in feat_cols])

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, name in zip(axes, ['MLP', 'Gradient Boosting']):
    s_ = perm[name].sort_values(ascending=False).head(8)[::-1]
    ax.hlines(s_.index, 0, s_.values, color=CORES[name], lw=2)
    ax.plot(s_.values, s_.index, 'o', color=CORES[name], ms=8)
    ax.set_title(name, fontweight='bold'); ax.set_xlabel('Queda média da ROC-AUC'); ax.set_xlim(0, s_.max() * 1.15)
    ax.grid(axis='y', alpha=0); limpa(ax)
plt.tight_layout(); plt.savefig('figs/importancia_permutacao.png', dpi=200); plt.close()

# %% [markdown]
# ## 9. Exportação dos resultados

# %%
feat_names = pipes['MLP'].named_steps['prep'].get_feature_names_out()
out = {'split': split_info, 'cv': cv_results, 'test': test_results,
       'n_features': int(len(feat_names)),
       'perm': {n: s.sort_values(ascending=False).round(4).to_dict() for n, s in perm.items()},
       'ap': ap, 'mlp_iters': int(pipes['MLP'].named_steps['clf'].n_iter_),
       'gb_iters': int(pipes['Gradient Boosting'].named_steps['clf'].n_iter_)}
with open('resultados.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('Total de atributos após transformação:', len(feat_names))
