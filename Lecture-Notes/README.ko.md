# 강의 노트

[레포지토리 안내](../README.ko.md) | [English](README.md) | [한국어](README.ko.md)

> **최종 수정일:** 2026-11-04

| 단원 | 주제 | 문서 |
|:-----|:-----|:-----|
| [1강 딥러닝 기초](Lec01_Deep-Learning-Fundamentals/) | 인공지능 응용 사례와 응용, 태스크, 모델의 관계, 지도학습, 비지도학습, 강화학습, 신경망의 구조와 전방 계산, 활성화 함수와 비선형성, 손실 함수, 경사하강법과 옵티마이저, 오류 역전파, 데이터셋 분할, 과적합과 과소적합, 배치 크기, 에포크와 이터레이션, 하이퍼파라미터, Colab, Jupyter, PyTorch, Docker, 사전 학습된 VGG-16을 이용한 이미지 분류 실습 | [강의 노트](Lec01_Deep-Learning-Fundamentals/Concepts_Lecture.ko.md), [실습 노트](Lec01_Deep-Learning-Fundamentals/Concepts_Lab.ko.md), [예제 노트북: 사전 학습된 VGG-16 불러오기](Lec01_Deep-Learning-Fundamentals/Example_Load-VGG.ipynb) |
| [2강 딥러닝 기초 2](Lec02_Deep-Learning-Fundamentals-2/) | 신경망 학습 과정의 복습, CNN의 합성곱과 커널, 패딩과 스트라이드, 다채널 입력, 풀링 층, AlexNet, 시계열 데이터와 RNN의 구조, LSTM과 GRU, PyTorch를 이용한 딥러닝 구현 흐름, PyTorch로 구현한 CNN과 LSTM 실습 | [강의 노트](Lec02_Deep-Learning-Fundamentals-2/Concepts_Lecture.ko.md), [실습 노트](Lec02_Deep-Learning-Fundamentals-2/Concepts_Lab.ko.md), [예제 노트북: CNN과 LSTM](Lec02_Deep-Learning-Fundamentals-2/Example_CNN-RNN.ipynb) |
| [3강 영상처리 1](Lec03_Image-Processing-1/) | PyTorch 구현 흐름(데이터셋, 데이터 로더, 모델, 손실 함수, 최적화 기법, 학습과 검증, 추론), 영상처리 태스크(이미지 분류, 객체 탐지, 분할, OCR, 초해상도, 이미지 생성)와 적용 사례, ImageNet 대회, VGGNet, ResNet과 잔차 블록, 기울기 소실과 폭주의 해결책, 전이 학습, CIFAR-10 분류 모델 학습과 간단한 ResNet 구현 실습 | [강의 노트](Lec03_Image-Processing-1/Concepts_Lecture.ko.md), [실습 노트](Lec03_Image-Processing-1/Concepts_Lab.ko.md), [예제 노트북: 학습 튜토리얼](Lec03_Image-Processing-1/Example_Training-Tutorial.ipynb), [예제 노트북: 간단한 ResNet](Lec03_Image-Processing-1/Example_ResNet-Simple.ipynb) |
| [4강 영상처리 2](Lec04_Image-Processing-2/) | VGGNet과 ResNet의 기여, 전이 학습과 미세 조정, 백본, 데이터 증강(Mixup, Cutout, CutMix, Mosaic), 혼동 행렬과 정확도, 정밀도, 재현율, 다중 클래스 F1 점수(macro, weighted, micro), 경진대회 기법(모델 선택, TTA, 보팅, 스태킹, model soups) | [강의 노트](Lec04_Image-Processing-2/Concepts_Lecture.ko.md) |
| [5강 영상처리 3](Lec05_Image-Processing-3/) | 객체 탐지의 분류와 박스 회귀, 2단계 탐지기와 1단계 탐지기, R-CNN과 영역 제안(슬라이딩 윈도, 선택적 탐색), Fast R-CNN과 Faster R-CNN, YOLO의 그리드 셀, 신뢰도, 손실 함수와 추론, NMS, YOLO의 발전, IoU와 정밀도-재현율 곡선, mAP | [강의 노트](Lec05_Image-Processing-3/Concepts_Lecture.ko.md) |
| [6강 소리처리](Lec06_Sound-Processing/) | 소리 데이터의 태스크(오디오 분류, 음성 인식, 음향 이벤트 탐지와 위치 추정, 오디오 생성과 잡음 제거), 소리 전처리(샘플링 레이트와 나이퀴스트 주파수, FFT, STFT, 스펙트로그램, 멜 스펙트로그램, MFCC), VGGish를 이용한 오디오 분류, 음향 이벤트 탐지(SED-CRNN, 세그먼트 기반 평가와 이벤트 기반 평가) | [강의 노트](Lec06_Sound-Processing/Concepts_Lecture.ko.md) |
| [7강 자연어 처리 1](Lec07_Natural-Language-Processing-1/) | 코퍼스, 토큰, 토큰화 등 기본 용어, 형태소 분석, 불용어, 어간 추출과 표제어 추출, 자연어 처리 과정과 응용, 단어 임베딩, 원-핫 인코딩, TF-IDF, Word2Vec(CBOW, Skip-gram, 네거티브 샘플링), FastText, RNN을 이용한 감성 분석 | [강의 노트](Lec07_Natural-Language-Processing-1/Concepts_Lecture.ko.md) |
| [8강 자연어 처리 2](Lec08_Natural-Language-Processing-2/) | seq2seq와 어텐션, Transformer, 셀프 어텐션의 쿼리, 키, 밸류, 스케일드 닷-프로덕트 어텐션, 멀티 헤드 어텐션, 잔차 연결과 층 정규화, 임베딩 층, BPE, 위치 인코딩, 마스크드 멀티 헤드 어텐션, BERT와 GPT, ViT | [강의 노트](Lec08_Natural-Language-Processing-2/Concepts_Lecture.ko.md) |
| [9강 경량 모델](Lec09_Lightweight-Models/) | 모델 경량화와 AI 엣지 컴퓨팅, ResNet의 병목 블록과 1×1 합성곱, SqueezeNet의 Fire 모듈, MobileNet의 깊이별 분리 합성곱과 폭 배수, 해상도 배수, ShuffleNet의 점별 그룹 합성곱과 채널 셔플 | [강의 노트](Lec09_Lightweight-Models/Concepts_Lecture.ko.md) |
