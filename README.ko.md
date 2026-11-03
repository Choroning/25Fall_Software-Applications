# [2025학년도 가을학기] 소프트웨어응용

![Last Commit](https://img.shields.io/github/last-commit/Choroning/25Fall_Software-Applications)
![Languages](https://img.shields.io/github/languages/top/Choroning/25Fall_Software-Applications)

이 레포지토리는 소프트웨어응용 과목의 이중 언어 강의 노트와 수업 결과물을 정리한 아카이브입니다.

*작성자: 박철원 (고려대학교(세종), 컴퓨터소프트웨어학과) - 2025년 기준 2학년*
<br><br>

## 📑 목차

- [레포지토리 소개](#about-this-repository)
- [강의 정보](#course-information)
- [사전 요구사항](#prerequisites)
- [레포지토리 구조](#repository-structure)
- [라이선스](#license)

---


<br><a name="about-this-repository"></a>
## 📝 레포지토리 소개

- `Lecture-Notes/`에는 강의자료 순서(`Lec01`, `Lec02` 등)를 따른 강의 노트가 있으며, 실습이 있었던 단원에는 실습 노트와 예제 노트북이 함께 있습니다.
- `Coursework/`에는 두 번의 논문 과제의 해설과 제출 보고서가 있습니다.
- 모든 강의 노트, 실습 노트, 과제 해설은 영어(`.md`)와 한국어(`.ko.md`) 파일이 짝을 이룹니다.

> **🤖 AI 에이전트 활용**
> [Claude Code](https://claude.ai/download)를 노트 정리와 번역, 레포지토리 정리에 활용했습니다.

<br><a name="course-information"></a>
## 📚 강의 정보

- **학기:** 2025학년도 가을학기 (9월 - 12월)
- **소속:** 고려대학교(세종)

|학수번호      |강의명    |이수구분|교수자|개설학과|
|:----------:|:-------|:----:|:------:|:----------------|
|`DCCS403-00`|소프트웨어응용|전공선택|채희찬 교수|컴퓨터소프트웨어학과|

### 과목 개요

영상, 소리, 텍스트 등 여러 소프트웨어 응용 분야에서 쓰이는 데이터 처리 기법과 딥러닝 모델을 배우고, 이론 강의에 이어 그 활용법을 실습하는 과목입니다. 교수자, 수업 일정, 평가, 수업 운영 규정, 진도는 [강의 정보 문서](Course-Information.ko.md)에 정리했습니다.

<br><a name="prerequisites"></a>
## ✅ 사전 요구사항

- Python 기초 프로그래밍 능력
- 기계학습 기초 개념에 대한 이해
- Google Colab을 사용할 수 있는 Google 계정

- **💻 개발 환경**

| 도구 | 회사 |  운영체제  | 비고 |
|:-----|:-------:|:----:|:------|
|Google Colab|Google|웹|실습 노트북 실행 환경|
|PyTorch|PyTorch Foundation|Linux (Colab)|딥러닝 프레임워크, torchvision 포함|

<br><a name="repository-structure"></a>
## 🗂 레포지토리 구조

```plaintext
25Fall_Software-Applications
├── Lecture-Notes               강의자료 순서의 강의 노트, 실습 노트, 예제 노트북
├── Coursework                  과제 해설과 제출 보고서
├── images                      강의 노트의 그림
├── Course-Information.ko.md    수업 운영 정보
├── Course-Information.md
├── LICENSE                     문서 라이선스 (CC BY-NC-SA 4.0)
├── LICENSE-CODE                코드 라이선스 (PolyForm Noncommercial 1.0.0)
├── README.ko.md
└── README.md
```

단원 목록은 [`Lecture-Notes` 색인](Lecture-Notes/README.ko.md)에, 과제는 [`Coursework` 색인](Coursework/README.ko.md)에 있습니다.

<br><a name="license"></a>
## 🤝 라이선스

이 레포지토리의 문서(강의 노트, 해설, 요약지)는 [CC BY-NC-SA 4.0](LICENSE), 코드는 [PolyForm Noncommercial 1.0.0](LICENSE-CODE)에 따라 배포됩니다. 강의자료와 교재에서 가져온 그림과 예제 코드, 강의가 배포한 과제 원본과 템플릿 코드는 원저작자에게 권리가 있으며 이 라이선스의 대상이 아닙니다.

---
