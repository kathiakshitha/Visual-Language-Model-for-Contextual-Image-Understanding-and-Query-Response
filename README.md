# Visual Question Answering (VQA) with BERT and ResNet50

This project implements a Visual Question Answering (VQA) system that combines a pre-trained ResNet50 for image feature extraction and a pre-trained BERT model for question understanding. The model answers questions about images using a sequence-to-sequence decoding approach with attention and beam search.

---

## 📌 Features

* **Image Feature Extraction** using ResNet50
* **Text Encoding** using BERT (`bert-base-uncased`)
* **Answer Generation** using a custom LSTM decoder with attention and beam search
* **Evaluation Metrics**: BLEU score for natural language generation quality
* **Data Augmentation** and preprocessing using torchvision
* **Train/Test split and DataLoaders** for efficient batch training

---

## 🗂️ Dataset

The dataset should include:

* `train_data.csv`, `eval_data.csv`: CSV files containing `image_id`, `question`, `response`
* A directory containing the actual image files
* The ResNet50 weights (`resnet50-11ad3fa6.pth`)

Dataset path examples:

```bash
/kaggle/input/vqa-dataset/archive (6)/train_data.csv
/kaggle/input/vqa-dataset/archive (6)/eval_data.csv
/kaggle/input/vqa-dataset/archive (8)/dataset/images
```

---

## 🛠 Requirements

Install the following Python packages:

```bash
pip install torch torchvision transformers pandas matplotlib pillow
```

This script also expects:

* CUDA-enabled GPU (optional but recommended)
* Kaggle-style directory paths or adjust paths manually for local use

---

## 🧠 Model Architecture

### 1. **CNN Feature Extractor**

* Uses **ResNet50**
* Removes final classification layer
* Maps 2048-D features → 512-D embedding

### 2. **BERT Question Encoder**

* Tokenizes question using `bert-base-uncased`
* Encodes using BERT’s `[CLS]` token
* Projects 768-D → 512-D

### 3. **Answer Decoder**

* LSTM decoder with 3 layers and dropout
* Uses an attention mechanism on combined image and question features
* Supports **beam search** decoding for inference

---

## 🚀 Training

To train the model:

```python
VQA_model = VQA_Model(answers_vocab_size)
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = AdamW(VQA_model.parameters(), lr=1e-4, weight_decay=1e-2)

VQA_model_history = train_model(
    VQA_model,
    train_loader,
    eval_loader,
    criterion,
    optimizer,
    '/kaggle/working/VAQ_model_bert.pth',
    num_epochs=50
)
```

Training includes:

* Early stopping based on evaluation loss
* BLEU score computation per epoch
* Saving the best model

---

## 🧪 Testing

```python
test_model(model, question, image_path, ground_truth, idx2word)
```

You can also run:

```python
test_random_samples(VQA_model, eval_dataframe, idx2word_answers)
```

To test on random 15 samples from the evaluation set and visualize results.

---

## 📊 Evaluation Metric

The model uses **BLEU Score** to evaluate the natural language quality of the generated answers.

---

## 🧾 Output Format

The answer decoder generates token sequences, which are decoded using the constructed vocabulary. Special tokens like `<sos>`, `<eos>`, `<pad>`, and `<unk>` are handled properly.

---

## 📁 Project Structure

```
.
├── vqa_with_bert.py
├── /dataset
│   ├── images/
│   ├── train_data.csv
│   ├── eval_data.csv
├── /resnet
│   └── resnet50-11ad3fa6.pth
└── README.md
```

---

## 💡 Notes

* The model uses a fixed vocabulary built from training answers.
* Questions are tokenized using BERT's tokenizer and truncated/padded to 24 tokens.
* Generated answers have a max length of 36 tokens.
* Beam search (`k=3`) improves answer generation quality over greedy decoding.

---

## 🧼 To Do / Improvements

* Integrate mixed precision training for speed
* Add support for more complex decoding strategies (e.g., top-k sampling)
* Improve answer vocabulary using subword tokenization (currently word-based)
* Visualize attention maps for interpretability

---

## 🤝 Acknowledgements

* [HuggingFace Transformers](https://huggingface.co/transformers/)
* [TorchVision Models](https://pytorch.org/vision/stable/models.html)
* Kaggle dataset contributors

---

Let me know if you'd like a Markdown version of this README or any adjustments (e.g., adapting it for a GitHub repo or Jupyter notebook).
