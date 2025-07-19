# Visual Question Answering (VQA) with BERT and ResNet50

This project implements a Visual Question Answering (VQA) system that combines a pre-trained ResNet50 for image feature extraction and a pre-trained BERT model for question understanding. The model answers questions about images using a sequence-to-sequence decoding approach with attention and beam search.

---

## 📌 Features

- **Image Feature Extraction** using ResNet50  
- **Text Encoding** using BERT (`bert-base-uncased`)  
- **Answer Generation** using a custom LSTM decoder with attention and beam search  
- **Evaluation Metrics**: BLEU score for natural language generation quality  
- **Data Augmentation** and preprocessing using torchvision  
- **Train/Test split and DataLoaders** for efficient batch training

---

## 🗂️ Dataset

The dataset should include:

- `train_data.csv`, `eval_data.csv`: CSV files containing `image_id`, `question`, `response`
- A directory containing the actual image files
- The ResNet50 weights (`resnet50-11ad3fa6.pth`)

Example dataset paths:

```
/kaggle/input/vqa-dataset/archive (6)/train_data.csv  
/kaggle/input/vqa-dataset/archive (6)/eval_data.csv  
/kaggle/input/vqa-dataset/archive (8)/dataset/images  
```

---

## 🛠 Requirements

Install the required packages:

```bash
pip install torch torchvision transformers pandas matplotlib pillow
```

Also required:

- CUDA-enabled GPU (optional but recommended)
- Kaggle-style directory structure or modify paths accordingly for local environments

---

## 🧠 Model Architecture

### 1. **CNN Feature Extractor**
- Uses **ResNet50**
- Removes the final classification layer
- Maps 2048-D → 512-D image feature embedding

### 2. **BERT Question Encoder**
- Tokenizes questions using `bert-base-uncased`
- Encodes using the `[CLS]` token representation
- Projects 768-D → 512-D question feature embedding

### 3. **Answer Decoder**
- LSTM decoder with 3 layers and dropout
- Uses attention over combined image + question embeddings
- Supports **beam search** decoding for inference

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

- Early stopping based on validation loss
- BLEU score computation at each epoch
- Best model saving based on performance

---

## 🧪 Testing

To test on a specific question and image:

```python
test_model(model, question, image_path, ground_truth, idx2word)
```

To test randomly on 15 samples from the evaluation set:

```python
test_random_samples(VQA_model, eval_dataframe, idx2word_answers)
```

This visualizes the image, question, and predicted answer.

---

## 📊 Evaluation Metric

- The model uses **BLEU Score** to assess the fluency and relevance of generated answers.

---

## 🧾 Output Format

- The decoder outputs token sequences.
- Tokens are converted to words using `idx2word` vocabulary.
- Handles `<sos>`, `<eos>`, `<pad>`, and `<unk>` tokens appropriately.

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

- The model uses a fixed vocabulary built from training answers.
- BERT tokenizes and pads/truncates questions to 24 tokens.
- Generated answers are capped at 36 tokens.
- Beam search (`k=3`) enhances answer generation compared to greedy decoding.

---

## 🧼 To Do / Improvements

- Integrate mixed precision training for faster training and less memory usage
- Implement advanced decoding (e.g., top-k or nucleus sampling)
- Use subword-level decoding instead of word-level
- Add attention visualization for interpretability

---

## 🤝 Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [TorchVision Models](https://pytorch.org/vision/stable/models.html)
- Kaggle dataset contributors
