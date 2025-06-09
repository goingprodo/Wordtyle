import os
import json
import pickle
import re
import random
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup,
    TrainingArguments, Trainer
)

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# 가속화 라이브러리들
try:
    import xformers
    import xformers.ops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StyleEmbeddingModel(nn.Module):
    """문체 임베딩을 위한 커스텀 모델"""
    
    def __init__(self, base_model_name: str = "klue/bert-base", 
                 embedding_dim: int = 384, num_style_classes: int = 8,
                 use_flash_attn: bool = False):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.base_model = AutoModel.from_pretrained(base_model_name)
        
        # 플래시 어텐션 설정
        if use_flash_attn and FLASH_ATTN_AVAILABLE:
            self.config.use_flash_attention_2 = True
            logger.info("Flash Attention 2 enabled")
        
        # 임베딩 프로젝션 레이어
        self.style_projector = nn.Sequential(
            nn.Linear(self.config.hidden_size, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # 문체 분류 헤드
        self.style_classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, num_style_classes)
        )
        
        self.embedding_dim = embedding_dim
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Base model forward
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # CLS 토큰 사용
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # 스타일 임베딩 생성
        style_embeddings = self.style_projector(pooled_output)
        
        # 문체 분류
        style_logits = self.style_classifier(style_embeddings)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(style_logits, labels)
        
        return {
            'loss': loss,
            'style_embeddings': style_embeddings,
            'style_logits': style_logits,
            'pooled_output': pooled_output
        }

class StyleDataset(Dataset):
    """문체 학습용 데이터셋"""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class StyleDataProcessor:
    """문체 데이터 전처리 클래스"""
    
    def __init__(self):
        self.style_patterns = {
            'formal': ['습니다', '입니다', '하였습니다', '되었습니다', '였습니다'],
            'informal': ['야', '어', '지', '잖아', '거야', '해'],
            'literary': ['처럼', '마치', '듯이', '것만 같았다', '스며들어', '흘러'],
            'dialogue': ['"', "'", '라고', '했다', '말했다', '물었다'],
            'narrative': ['그는', '그녀는', '이때', '그 순간', '한편', '그러나'],
            'poetic': ['달빛', '바람', '꽃잎', '별', '구름', '노을'],
            'technical': ['시스템', '데이터', '분석', '결과', '방법', '과정'],
            'emotional': ['가슴', '마음', '눈물', '기쁨', '슬픔', '분노']
        }
        
        self.label_map = {
            'formal': 0, 'informal': 1, 'literary': 2, 'dialogue': 3,
            'narrative': 4, 'poetic': 5, 'technical': 6, 'emotional': 7
        }
        
    def extract_sentences(self, text: str) -> List[str]:
        """텍스트에서 문장 추출"""
        # 문장 분리
        sentences = re.split(r'[.!?]\s*', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences
    
    def classify_style(self, sentence: str) -> int:
        """문장의 문체 분류"""
        sentence = sentence.lower()
        style_scores = {}
        
        for style, patterns in self.style_patterns.items():
            score = sum(1 for pattern in patterns if pattern in sentence)
            style_scores[style] = score
        
        # 가장 높은 점수의 스타일 선택
        if max(style_scores.values()) == 0:
            return self.label_map['narrative']  # 기본값
        
        best_style = max(style_scores, key=style_scores.get)
        return self.label_map[best_style]
    
    def create_dataset(self, book_text: str, min_sentence_length: int = 20) -> Tuple[List[str], List[int]]:
        """책 텍스트로부터 문체 데이터셋 생성"""
        sentences = self.extract_sentences(book_text)
        
        # 길이 필터링
        sentences = [s for s in sentences if len(s) >= min_sentence_length]
        
        # 문체 분류
        texts = []
        labels = []
        
        for sentence in sentences:
            label = self.classify_style(sentence)
            texts.append(sentence)
            labels.append(label)
        
        # 데이터 증강: 문장 조합
        augmented_texts, augmented_labels = self.augment_data(texts, labels)
        texts.extend(augmented_texts)
        labels.extend(augmented_labels)
        
        return texts, labels
    
    def augment_data(self, texts: List[str], labels: List[int], 
                    augment_ratio: float = 0.3) -> Tuple[List[str], List[int]]:
        """데이터 증강"""
        augmented_texts = []
        augmented_labels = []
        
        num_augment = int(len(texts) * augment_ratio)
        
        for _ in range(num_augment):
            # 같은 스타일의 문장들 조합
            idx1, idx2 = random.sample(range(len(texts)), 2)
            if labels[idx1] == labels[idx2]:
                combined_text = f"{texts[idx1]} {texts[idx2]}"
                augmented_texts.append(combined_text)
                augmented_labels.append(labels[idx1])
        
        return augmented_texts, augmented_labels

class StyleEmbeddingTrainer:
    """문체 임베딩 모델 트레이너"""
    
    def __init__(self, model_name: str = "klue/bert-base", 
                 output_dir: str = "style_models"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.data_processor = StyleDataProcessor()
        
        # 가속화 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mixed_precision = torch.cuda.is_available()
        self.use_xformers = XFORMERS_AVAILABLE and torch.cuda.is_available()
        self.use_flash_attn = FLASH_ATTN_AVAILABLE and torch.cuda.is_available()
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed Precision: {self.use_mixed_precision}")
        logger.info(f"XFormers: {self.use_xformers}")
        logger.info(f"Flash Attention: {self.use_flash_attn}")
        
        os.makedirs(output_dir, exist_ok=True)
        
    def prepare_data(self, book_text: str, test_size: float = 0.2) -> Dict:
        """데이터 준비"""
        logger.info("Processing book text...")
        texts, labels = self.data_processor.create_dataset(book_text)
        
        # 학습/검증 분할
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # 데이터셋 생성
        train_dataset = StyleDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = StyleDataset(val_texts, val_labels, self.tokenizer)
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        # 라벨 분포 확인
        unique_labels, counts = np.unique(labels, return_counts=True)
        logger.info("Label distribution:")
        for label, count in zip(unique_labels, counts):
            style_name = [k for k, v in self.data_processor.label_map.items() if v == label][0]
            logger.info(f"  {style_name} ({label}): {count}")
        
        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'train_texts': train_texts,
            'val_texts': val_texts,
            'train_labels': train_labels,
            'val_labels': val_labels
        }
        
    def create_model(self, num_style_classes: int = 8, embedding_dim: int = 384):
        """모델 생성"""
        self.model = StyleEmbeddingModel(
            base_model_name=self.model_name,
            embedding_dim=embedding_dim,
            num_style_classes=num_style_classes,
            use_flash_attn=self.use_flash_attn
        )
        
        if self.use_xformers:
            # XFormers 최적화 적용
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info("XFormers optimization applied")
        
        self.model.to(self.device)
        return self.model
        
    def train_model(self, data_dict: Dict, num_epochs: int = 3, 
                   batch_size: int = 16, learning_rate: float = 2e-5,
                   progress_callback=None) -> Dict:
        """모델 훈련"""
        
        if self.model is None:
            self.create_model()
            
        train_dataset = data_dict['train_dataset']
        val_dataset = data_dict['val_dataset']
        
        # 데이터 로더
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available()
        )
        
        # 옵티마이저와 스케줄러
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        # Mixed precision 설정
        scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        
        # 훈련 루프
        training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        self.model.train()
        for epoch in range(num_epochs):
            epoch_train_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # 데이터를 GPU로 이동
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids, attention_mask, labels)
                        loss = outputs['loss']
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(input_ids, attention_mask, labels)
                    loss = outputs['loss']
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                
                epoch_train_loss += loss.item()
                num_batches += 1
                
                # 진행률 업데이트
                if progress_callback:
                    progress = ((epoch * len(train_loader) + batch_idx + 1) / 
                              (num_epochs * len(train_loader))) * 100
                    progress_callback(int(progress))
            
            # 검증
            val_loss, val_accuracy = self.evaluate_model(val_loader)
            
            avg_train_loss = epoch_train_loss / num_batches
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['val_accuracy'].append(val_accuracy)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Val Accuracy: {val_accuracy:.4f}")
        
        return training_history
    
    def evaluate_model(self, val_loader):
        """모델 평가"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids, attention_mask, labels)
                else:
                    outputs = self.model(input_ids, attention_mask, labels)
                
                total_loss += outputs['loss'].item()
                
                predictions = torch.argmax(outputs['style_logits'], dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        self.model.train()
        return avg_loss, accuracy
    
    def save_model(self, model_name: str = None) -> str:
        """모델 저장"""
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"style_embedding_model_{timestamp}"
        
        model_path = os.path.join(self.output_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        # 모델 저장
        torch.save(self.model.state_dict(), os.path.join(model_path, "pytorch_model.bin"))
        
        # 토크나이저 저장
        self.tokenizer.save_pretrained(model_path)
        
        # 설정 저장
        config = {
            'model_name': self.model_name,
            'embedding_dim': self.model.embedding_dim,
            'num_style_classes': len(self.data_processor.label_map),
            'label_map': self.data_processor.label_map,
            'style_patterns': self.data_processor.style_patterns
        }
        
        with open(os.path.join(model_path, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Model saved to: {model_path}")
        return model_path
    
    def extract_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """텍스트에서 임베딩 추출"""
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                encoding = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids, attention_mask)
                else:
                    outputs = self.model(input_ids, attention_mask)
                
                batch_embeddings = outputs['style_embeddings'].cpu().numpy()
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)

def create_training_gui():
    """문체 임베딩 모델 훈련용 Gradio 인터페이스 생성"""
    
    # 전역 상태 관리
    trainer = None
    book_text_content = None
    
    def process_book_file(file_path, min_sentence_length):
        """책 파일 처리"""
        nonlocal book_text_content
        
        if file_path is None:
            return None, "파일을 선택해주세요."
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                book_text_content = f.read()
            
            # 텍스트 통계
            sentences = StyleDataProcessor().extract_sentences(book_text_content)
            sentences = [s for s in sentences if len(s) >= min_sentence_length]
            
            stats = f"""
📊 **텍스트 분석 결과:**
- 총 문자 수: {len(book_text_content):,}
- 추출된 문장 수: {len(sentences):,}
- 평균 문장 길이: {np.mean([len(s) for s in sentences]):.1f}자
- 최소 문장 길이: {min_sentence_length}자 이상

✅ 파일 처리 완료! 이제 모델 훈련을 시작할 수 있습니다.
            """
            
            return book_text_content, stats
            
        except Exception as e:
            book_text_content = None
            return None, f"파일 처리 중 오류 발생: {str(e)}"
    
    def train_embedding_model(book_text, model_name, num_epochs, batch_size, 
                            learning_rate, embedding_dim, progress=gr.Progress()):
        """임베딩 모델 훈련"""
        nonlocal trainer
        
        if book_text is None or not book_text:
            return "먼저 책 파일을 업로드하고 처리해주세요.", "", ""
        
        try:
            # 트레이너 초기화
            trainer = StyleEmbeddingTrainer(model_name=model_name)
            
            # 진행률 콜백
            def update_progress(value):
                progress(value / 100, desc=f"모델 훈련 중... {value}%")
            
            # 데이터 준비
            progress(10, desc="데이터 준비 중...")
            data_dict = trainer.prepare_data(book_text)
            
            # 모델 생성
            progress(20, desc="모델 생성 중...")
            trainer.create_model(embedding_dim=embedding_dim)
            
            # 훈련 시작
            progress(30, desc="훈련 시작...")
            history = trainer.train_model(
                data_dict=data_dict,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                progress_callback=update_progress
            )
            
            # 모델 저장
            progress(95, desc="모델 저장 중...")
            model_path = trainer.save_model()
            
            progress(100, desc="완료!")
            
            # 훈련 결과 요약
            final_train_loss = history['train_loss'][-1]
            final_val_loss = history['val_loss'][-1]
            final_val_acc = history['val_accuracy'][-1]
            
            result_summary = f"""
🎉 **훈련 완료!**

📈 **최종 결과:**
- 훈련 손실: {final_train_loss:.4f}
- 검증 손실: {final_val_loss:.4f}
- 검증 정확도: {final_val_acc:.4f}

💾 **모델 저장 경로:** {model_path}
"""
            
            # 훈련 과정 데이터
            history_text = "Epoch\tTrain Loss\tVal Loss\tVal Accuracy\n"
            for i, (tl, vl, va) in enumerate(zip(history['train_loss'], 
                                               history['val_loss'], 
                                               history['val_accuracy'])):
                history_text += f"{i+1}\t{tl:.4f}\t{vl:.4f}\t{va:.4f}\n"
            
            return result_summary, history_text, model_path
            
        except Exception as e:
            return f"훈련 중 오류 발생: {str(e)}", "", ""
    
    def test_embeddings(model_path, test_text):
        """임베딩 테스트"""
        nonlocal trainer
        
        if trainer is None or not model_path:
            return "먼저 모델을 훈련해주세요."
        
        if not test_text or not test_text.strip():
            return "테스트할 텍스트를 입력해주세요."
        
        try:
            # 테스트 텍스트 임베딩 추출
            embeddings = trainer.extract_embeddings([test_text])
            
            # 스타일 예측
            trainer.model.eval()
            with torch.no_grad():
                encoding = trainer.tokenizer(
                    test_text,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(trainer.device)
                attention_mask = encoding['attention_mask'].to(trainer.device)
                
                outputs = trainer.model(input_ids, attention_mask)
                style_probs = F.softmax(outputs['style_logits'], dim=-1)
                style_probs = style_probs.cpu().numpy()[0]
            
            # 결과 포맷팅
            label_map = trainer.data_processor.label_map
            reverse_label_map = {v: k for k, v in label_map.items()}
            
            result = "🎯 **스타일 분석 결과:**\n\n"
            for i, prob in enumerate(style_probs):
                style_name = reverse_label_map[i]
                result += f"- {style_name}: {prob:.3f} ({prob*100:.1f}%)\n"
            
            result += f"\n📊 **임베딩 차원:** {embeddings.shape[1]}"
            
            return result
            
        except Exception as e:
            return f"테스트 중 오류: {str(e)}"

    # Gradio 인터페이스 구성
    with gr.Blocks(title="Style Embedding Model Trainer") as demo:
        gr.Markdown("# 🎨 문체 임베딩 모델 트레이너")
        gr.Markdown("책 한 권을 업로드하여 문체 임베딩 모델을 훈련하세요!")
        
        with gr.Tabs():
            with gr.TabItem("📚 데이터 준비"):
                gr.Markdown("### 1. 책 파일 업로드")
                book_file = gr.File(
                    label="텍스트 파일 (.txt)",
                    file_types=[".txt"]
                )
                
                min_sentence_length = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=20,
                    step=5,
                    label="최소 문장 길이 (문자)"
                )
                
                process_btn = gr.Button("📊 파일 분석", variant="primary")
                
                book_stats = gr.Markdown()
                book_text_state = gr.State()
                
            with gr.TabItem("🏋️ 모델 훈련"):
                gr.Markdown("### 훈련 설정")
                
                with gr.Row():
                    with gr.Column():
                        model_name = gr.Dropdown(
                            choices=[
                                "klue/bert-base",
                                "klue/roberta-base", 
                                "beomi/KcELECTRA-base",
                                "monologg/kobert"
                            ],
                            value="klue/bert-base",
                            label="베이스 모델"
                        )
                        
                        num_epochs = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="에포크 수"
                        )
                        
                        batch_size = gr.Slider(
                            minimum=4,
                            maximum=64,
                            value=16,
                            step=4,
                            label="배치 크기"
                        )
                    
                    with gr.Column():
                        learning_rate = gr.Number(
                            value=2e-5,
                            label="학습률"
                        )
                        
                        embedding_dim = gr.Slider(
                            minimum=128,
                            maximum=768,
                            value=384,
                            step=64,
                            label="임베딩 차원"
                        )
                
                train_btn = gr.Button("🚀 훈련 시작", variant="primary", size="lg")
                
                training_result = gr.Markdown()
                training_history = gr.Textbox(
                    label="훈련 과정",
                    lines=10,
                    interactive=False
                )
                model_path_state = gr.State()
                
            with gr.TabItem("🧪 모델 테스트"):
                gr.Markdown("### 임베딩 테스트")
                
                test_text = gr.Textbox(
                    label="테스트 텍스트",
                    placeholder="문체를 분석할 텍스트를 입력하세요...",
                    lines=3,
                    value="그는 조용히 문을 열고 방 안으로 들어갔다."
                )
                
                test_btn = gr.Button("🔍 분석하기", variant="secondary")
                
                test_result = gr.Markdown()
        
        # ⚠️ 핵심: 이벤트 핸들러들 - 이 부분이 없으면 버튼이 작동하지 않습니다!
        process_btn.click(
            fn=process_book_file,
            inputs=[book_file, min_sentence_length],
            outputs=[book_text_state, book_stats]
        )
        
        train_btn.click(
            fn=train_embedding_model,
            inputs=[
                book_text_state, model_name, num_epochs, batch_size,
                learning_rate, embedding_dim
            ],
            outputs=[training_result, training_history, model_path_state],
            show_progress=True
        )
        
        test_btn.click(
            fn=test_embeddings,
            inputs=[model_path_state, test_text],
            outputs=test_result
        )
        
        # 가속화 정보
        gr.Markdown("## ⚡ 가속화 상태")
        acceleration_info = f"""
        - **CUDA:** {'✅' if torch.cuda.is_available() else '❌'}
        - **XFormers:** {'✅' if XFORMERS_AVAILABLE else '❌'}
        - **Flash Attention 2:** {'✅' if FLASH_ATTN_AVAILABLE else '❌'}
        """
        gr.Markdown(acceleration_info)
        
        # 사용 가이드
        gr.Markdown("""
        ## 📖 사용 방법
        
        1. **📚 데이터 준비** 탭에서 `.txt` 파일을 업로드하고 '📊 파일 분석' 버튼을 클릭
        2. **🏋️ 모델 훈련** 탭에서 설정을 조정하고 '🚀 훈련 시작' 버튼을 클릭  
        3. **🧪 모델 테스트** 탭에서 훈련된 모델을 테스트
        
        ### 💡 팁
        - 최소 10,000자 이상의 텍스트 권장
        - GPU가 있으면 배치 크기를 16-32로 설정
        - CPU만 있으면 배치 크기를 4-8로 설정
        
        ### 🚨 문제 해결
        - 버튼을 눌러도 반응이 없다면 콘솔(F12)에서 에러를 확인하세요
        - 파일 업로드 후 반드시 '📊 파일 분석' 버튼을 눌러주세요
        - 메모리 부족 시 배치 크기를 줄여주세요
        """)
    
    return demo

if __name__ == "__main__":
    print("🔍 가속화 라이브러리 확인:")
    print(f"  - PyTorch CUDA: {'✅' if torch.cuda.is_available() else '❌'}")
    print(f"  - XFormers: {'✅' if XFORMERS_AVAILABLE else '❌'}")
    print(f"  - Triton: {'✅' if TRITON_AVAILABLE else '❌'}")
    print(f"  - Flash Attention 2: {'✅' if FLASH_ATTN_AVAILABLE else '❌'}")
    
    if torch.cuda.is_available():
        print(f"  - GPU: {torch.cuda.get_device_name()}")
        print(f"  - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    print("🚀 Gradio 인터페이스 시작 중...")
    
    # 실제 실행 코드
    demo = create_training_gui()
    demo.launch(
        server_name="127.0.0.1",  # localhost로 변경
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )
    
    print("✅ 문체 임베딩 트레이너가 시작되었습니다!")
    print("🌐 브라우저에서 http://localhost:7860 로 접속하세요!")