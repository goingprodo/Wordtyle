# Wordtyle - Korean Writing Style Embedding Model Trainer

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-orange.svg)
![License](https://img.shields.io/badge/license-AGPL%20v3-green.svg)

Wordtyle is a deep learning tool for training Korean writing style embedding models. It analyzes text writing styles and converts them into embedding vectors. With its user-friendly Gradio-based web interface, anyone can easily create writing style analysis models.

## ✨ Key Features

- 📚 **Book Text-based Training**: Upload text files (novels, essays, etc.) to train writing style analysis models
- 🎯 **8 Writing Style Classifications**: Automatic classification of formal, informal, literary, dialogue, narrative, poetic, technical, and emotional styles
- 🚀 **GPU Acceleration Support**: High-speed training with CUDA, XFormers, and Flash Attention 2
- 🎨 **Intuitive Web Interface**: Easy-to-use Gradio-based GUI
- 💾 **Model Saving & Reusability**: Save trained models and load them for later use
- 🧪 **Real-time Testing**: Instantly test text style analysis after training completion

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/wordtyle.git
cd wordtyle
```

### 2. Create Virtual Environment

```bash
python -3.12 -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

#### Basic Installation (CPU only)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

#### GPU Installation (Recommended)
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install basic requirements
pip install -r requirements.txt

# Optional: Install acceleration libraries
pip install xformers>=0.0.20  # Memory optimization
pip install triton>=2.0.0     # CUDA kernel optimization
pip install flash-attn>=2.0.0 # Flash Attention 2
```

## 🚀 Quick Start

### 1. Launch the Application

```bash
python main.py
```

Or use the batch file on Windows:
```bash
run_gpu.bat
```

### 2. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:7860
```

### 3. Train Your Model

1. **📚 Data Preparation Tab**:
   - Upload a Korean text file (.txt)
   - Set minimum sentence length (recommended: 20+ characters)
   - Click "📊 Analyze File"

2. **🏋️ Model Training Tab**:
   - Configure training parameters:
     - Base model (default: klue/bert-base)
     - Number of epochs (1-10)
     - Batch size (adjust based on your GPU memory)
     - Learning rate (default: 2e-5)
     - Embedding dimension (128-768)
   - Click "🚀 Start Training"

3. **🧪 Model Testing Tab**:
   - Enter test text to analyze writing style
   - View style probability distributions
   - Get embedding vectors

## 📊 Writing Style Categories

The model classifies text into 8 different Korean writing styles:

| Style | Korean | Description | Example Patterns |
|-------|--------|-------------|------------------|
| **Formal** | 격식체 | Formal/polite language | 습니다, 입니다, 하였습니다 |
| **Informal** | 비격식체 | Casual/informal language | 야, 어, 지, 잖아, 거야 |
| **Literary** | 문학적 | Literary/artistic style | 처럼, 마치, 듯이, 것만 같았다 |
| **Dialogue** | 대화체 | Conversational style | ", ', 라고, 했다, 말했다 |
| **Narrative** | 서술체 | Narrative/descriptive | 그는, 그녀는, 이때, 그 순간 |
| **Poetic** | 시적 | Poetic/lyrical style | 달빛, 바람, 꽃잎, 별, 구름 |
| **Technical** | 기술적 | Technical/analytical | 시스템, 데이터, 분석, 결과 |
| **Emotional** | 감성적 | Emotional/expressive | 가슴, 마음, 눈물, 기쁨, 슬픔 |

## 🔧 Configuration

### Model Parameters

- **Base Model**: Choose from pre-trained Korean language models
  - `klue/bert-base` (default)
  - `klue/roberta-base`
  - `beomi/KcELECTRA-base`
  - `monologg/kobert`

- **Training Parameters**:
  - Epochs: 1-10 (default: 3)
  - Batch Size: 4-64 (adjust based on GPU memory)
  - Learning Rate: 1e-6 to 1e-4 (default: 2e-5)
  - Embedding Dimension: 128-768 (default: 384)

### Hardware Requirements

#### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 2GB free space

#### Recommended Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 5GB+ free space

### Performance Optimization

The application automatically detects and uses available acceleration libraries:

- ✅ **CUDA**: GPU acceleration
- ✅ **XFormers**: Memory-efficient attention
- ✅ **Flash Attention 2**: Faster attention computation
- ✅ **Mixed Precision**: Reduced memory usage

## 📁 Project Structure

```
wordtyle/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── run_gpu.bat         # Windows batch file for easy execution
├── README.md           # This file
├── style_models/       # Directory for saved models
├── example/            # Example files and demos
└── venv/              # Virtual environment (after setup)
```

## 🎯 Usage Examples

### Basic Training Example

```python
from main import StyleEmbeddingTrainer

# Initialize trainer
trainer = StyleEmbeddingTrainer(model_name="klue/bert-base")

# Load your book text
with open("my_novel.txt", "r", encoding="utf-8") as f:
    book_text = f.read()

# Prepare data and train
data_dict = trainer.prepare_data(book_text)
trainer.create_model(embedding_dim=384)
history = trainer.train_model(data_dict, num_epochs=3)

# Save the model
model_path = trainer.save_model("my_style_model")
```

### Extract Embeddings

```python
# Extract style embeddings from text
texts = ["그는 조용히 문을 열고 방 안으로 들어갔다.", 
         "야, 뭐하고 있어?"]
embeddings = trainer.extract_embeddings(texts)
print(f"Embedding shape: {embeddings.shape}")
```

## 🔍 Troubleshooting

### Common Issues

1. **Buttons not responding**
   - Check browser console (F12) for errors
   - Make sure to click "📊 Analyze File" after uploading

2. **Out of memory errors**
   - Reduce batch size (try 4-8 for CPU, 8-16 for GPU)
   - Reduce embedding dimension
   - Use smaller text files

3. **Slow training**
   - Install GPU acceleration libraries
   - Increase batch size if memory allows
   - Use smaller models (bert-base instead of bert-large)

4. **Import errors**
   - Make sure virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt`
   - For GPU: Install PyTorch with CUDA support

### Performance Tips

- **For CPU training**: Use batch size 4-8, embedding dim 256
- **For GPU training**: Use batch size 16-32, embedding dim 384-512
- **Text size**: Minimum 10,000 characters recommended for good results
- **Sentence length**: Set minimum 20 characters for quality data

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/yourusername/wordtyle.git
cd wordtyle
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .
```

## 📄 License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

This means:
- ✅ You can use, modify, and distribute this software
- ✅ You can use it for commercial purposes
- ❗ If you run this software on a server and provide it as a service, you must make the source code available to users
- ❗ Any modifications must also be licensed under AGPL-3.0
- ❗ You must include the original license and copyright notice

See the [LICENSE](LICENSE) file for full details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the transformer models
- [KLUE](https://klue-benchmark.com/) for Korean language understanding models
- [Gradio](https://gradio.app/) for the web interface framework
- [PyTorch](https://pytorch.org/) for the deep learning framework

## 📞 Support

- 🐛 **Bug Reports**: [Open an issue](https://github.com/yourusername/wordtyle/issues)
- 💡 **Feature Requests**: [Open an issue](https://github.com/yourusername/wordtyle/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/wordtyle/discussions)

## 🔗 Related Projects

- [KoBERT](https://github.com/SKTBrain/KoBERT): Korean BERT model
- [Sentence Transformers](https://www.sbert.net/): Sentence embedding models
- [KLUE](https://github.com/KLUE-benchmark/KLUE): Korean Language Understanding Evaluation

---

Made with ❤️ for the Korean NLP community
