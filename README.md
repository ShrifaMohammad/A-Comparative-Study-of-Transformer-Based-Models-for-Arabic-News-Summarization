# A Comparative Study of Transformer-Based Models for Arabic News Summarization

A rigorous empirical comparison of three transformer architectures — **mT5-small**, **AraT5-base**, and **AraBART** — for abstractive Arabic news summarization, evaluating both summary quality and production-readiness.

> Prepared for Dr. Areej Alshutairy

## 📄 Full Paper

The complete research paper — including literature review, detailed methodology, and full results discussion — is available here: [`Arabic_Summarization_Paper.pdf`](Arabic_Summarization_Paper.pdf)

## Key Result

**AraBART (Arabic-specific BART) decisively outperformed both alternatives**, while also being the fastest:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Inference time/sample |
|---|---|---|---|---|
| mT5-small (multilingual) | 0.087 | 0.024 | 0.087 | 6.23s |
| AraT5-base | 0.001 | 0.000 | 0.001 | 7.85s |
| **AraBART** | **0.454** | **0.194** | **0.445** | **2.00s** |

- **4.2× improvement** in ROUGE-1 and **3.1× faster inference** vs. the multilingual mT5-small baseline
- AraT5-base suffered a catastrophic generation failure — despite a normal training loss curve, its outputs were largely non-Arabic, repetitive tokens, illustrating that **training loss alone is not a reliable signal of model quality**

## Research questions

1. Which transformer architecture performs best for Arabic news summarization?
2. Do Arabic-specific pre-trained models outperform multilingual alternatives?
3. What are the quality–efficiency trade-offs for production deployment?

## Dataset

[`Abdelkareem/Arabic-article-summarization-30-000`](https://huggingface.co/datasets/Abdelkareem/Arabic-article-summarization-30-000) (Hugging Face) — 5,000 Arabic news articles with human-written summaries, spanning international news, local news, and sports.

- Mean article length: 111.4 words · Mean summary length: 42.1 words
- Compression ratio: ~3.36:1
- Split: 80% train (4,000) / 10% validation (500) / 10% test (500), stratified by topic

## Models compared

| Model | Params | Pre-training | Role |
|---|---|---|---|
| mT5-small | 300M | mC4, 101 languages, span-corruption | Multilingual baseline |
| AraT5-base | 220M | 70GB Arabic corpora, span-corruption | Arabic-specific T5 |
| AraBART | 139M | 61GB Arabic text, denoising (masking + deletion + permutation + rotation) | Task-aligned Arabic generation model |

## Methodology

1. **Six-stage preprocessing pipeline**: text normalization (diacritics, letter variants, tatweel removal), special character handling, Arabic stopword removal (NLTK, 246 stopwords), model-specific tokenization, quality filtering (removed 47 erroneous pairs, 12 duplicates), stratified train/val/test split
2. **Identical training configuration across all 3 models** for fair comparison: AdamW, learning rate 5e-5, effective batch size 16, 2 epochs, FP16 mixed precision, on a Tesla T4 GPU (Google Colab)
3. **Evaluation**: ROUGE-1 / ROUGE-2 / ROUGE-L (F1) for quality, wall-clock inference time (beam search, beam size 4) for efficiency
4. **Extended analysis**: Named Entity Recognition (CAMeL-Lab Arabic BERT NER) and TF-IDF keyphrase extraction on generated summaries, to evaluate semantic content preservation beyond ROUGE scores — AraBART summaries preserved **92.9%** of reference named entities

## Why AraBART won

- Its denoising pre-training objective (token masking/deletion/permutation) is naturally aligned with summarization, unlike mT5's general span-corruption objective
- Smaller parameter count (139M vs. 300M) and a more efficient Arabic-tokenizer (65K vocabulary with morphological awareness) made it both more accurate and faster
- Smooth, stable training loss convergence (0.127 → 0.051) vs. mT5's steep adaptation curve (26.6 → 6.3), suggesting much better initialization for the Arabic generation task

## Limitations

- Single dataset/domain (news only) — results may not generalize across dialects or text types
- Models trained for only 2 epochs due to compute constraints
- ROUGE metrics don't directly capture factual accuracy or semantic equivalence; human evaluation would add complementary insight

## Tech stack

- Hugging Face `transformers`, `datasets`
- PyTorch
- NLTK (Arabic stopwords)
- ROUGE-score, scikit-learn (TF-IDF)
- CAMeL-Lab Arabic BERT (NER)
- Matplotlib, Seaborn, Plotly

## Repo contents

```
├── arabic_summarization.ipynb        # Full pipeline: EDA, preprocessing, training, evaluation, NER & keyphrase analysis
├── Arabic_Summarization_Paper.pdf    # Full research paper
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/arabic-news-summarization.git
cd arabic-news-summarization
pip install -r requirements.txt
```

Open `arabic_summarization.ipynb` in Jupyter or Google Colab (a GPU runtime is strongly recommended — training each model took ~2–2.5 hours on a Tesla T4). The dataset loads automatically from Hugging Face via `datasets.load_dataset`.

## Future work

- Multi-task learning incorporating NER and keyphrase extraction as auxiliary training objectives
- Cross-domain evaluation (scientific articles, legal text, social media)
- Human evaluation of fluency, coherence, and factual accuracy
- Dialect-specific models (Egyptian, Levantine, Gulf, Maghrebi)
- Investigating the AraT5-base failure mode through ablation studies

## Team

| Member |
|---|
| Sharifa Mohammed |
| Hanan Awdh |
| Raghad Bawazeer |
| Haneen Bamaas |
| Mais Mohammed |

## References

Full citation list available in the [paper](Arabic_Summarization_Paper.pdf).
