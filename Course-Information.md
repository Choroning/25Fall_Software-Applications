# Course Information

[Repository Guide](README.md) | [English](Course-Information.md) | [한국어](Course-Information.ko.md)

> **Last Updated:** 2026-11-04

## Table of Contents

- [1. Instructor and Research Lab](#1-instructor-and-research-lab)
- [2. Schedule and Class Format](#2-schedule-and-class-format)
- [3. Assessment](#3-assessment)
- [4. Course Policies](#4-course-policies)
- [5. Weekly Plan and Progress](#5-weekly-plan-and-progress)
- [6. References](#6-references)

---

<br>

## 1. Instructor and Research Lab

- **Instructor:** Prof. Heechan Chae (채희찬)
- **Office:** Room 514, Industry-Academic Cooperation Building
- **Office Hours:** Weekdays 09:00 ~ 17:00

---

<br>

## 2. Schedule and Class Format

- **Credits (Hours):** 3 (3)
- **Intended For:** Fourth-year students (major elective)
- **Class Time:** Tuesday periods 5-7 (13:00 ~ 15:50)
- **Classroom:** Room 7-315
- **Class Format:** The activity types in the syllabus are lectures, presentations, and practice. Each theory lecture on deep learning applications is followed by practice, an individual project proposal based on the lectures is presented at the end of the semester, and classes are held face to face in principle.
- **Design Goals:** The course consists of lectures with practice and presentations.
  - Lectures and practice: Lectures are based on the lecture slides and proceed not only as one-way delivery of knowledge but also interactively through questions and answers with the students. Practice based on the lecture content teaches practical as well as theoretical content.
  - Presentations: Students design and concretize a deep learning application of their own based on the lectures and practice. They improve their problem-solving skills by examining the problems or weaknesses of a project or program they are interested in and proposing improvements, and they broaden their understanding of research topics by seeing the case studies of other students.
- **Attendance:** Attendance is checked (the course is not exempt from attendance checks).
- **Exam Proctoring:** Exams are not held without proctoring.
- **Course Description:** Deep learning is now widely used in various software application fields such as images, sound, and text. Going beyond theory, students learn the data processing techniques and deep learning models used in practice in various application fields and practice applying them.
- **Learning Goals:**
  1. Acquire various data processing skills.
  2. Learn and use applied deep learning models.
  3. Build the ability to use open datasets.
  4. Build the ability to design deep learning applications creatively.
- **Link Between the Core Competencies, Department Goals, and Course Goals:** Among the major competencies of the department, Think is linked to building creativity through the proposal presentation, Technology to acquiring data processing skills, and Application to improving the ability to use deep learning applications.
- **Core Competencies:** The syllabus marks practical application (실무실용) and creative convergence (창의융합) as very closely related competencies.
- **Major Competencies:** Think & Technology, Network & Application
- **Prerequisites:** The recommended prerequisites and enrollment requirements are basic Python practice skills and basic machine learning concepts.
- **Lab Environment:** Practice sessions used Jupyter notebooks written with PyTorch and run on Google Colab.

---

<br>

## 3. Assessment

| Item | Weight |
|:-----|-----:|
| Quizzes (assignments and presentation) | 25% |
| Midterm Exam | 30% |
| Final Exam | 35% |
| Participation | 10% |

- Assessment scores are disclosed.
- **Assessment Plan:** The quiz component consists of assignments and a presentation, midterm and final exams are held, and attendance and participation are scored.
- The weeks in which assignments were given are listed in [5. Weekly Plan and Progress](#5-weekly-plan-and-progress). The assignment explanations and the submitted report are in the [`Coursework` index](Coursework/README.md).
- **Assignment 1 (survey of image processing (deep learning) research papers):** Given in Week 5 and due 2025-10-17 (Fri) 23:59.
  - Summarize the ResNet paper (K. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016).
  - Choose one image processing task other than image classification, describe it including its input and output, and briefly give an example of where the task is used (e.g., road area detection in self-driving cars).
  - Choose one international paper on that task (cited 100 times or more, with a screenshot of the citation count) and describe the contribution of the research.
  - Include references without fail (using Google Scholar).
  - The length is about 4 pages (Word or PDF), in a free format.
- **Assignment 2 (GPT paper summary):** Given in Week 11 and due 2025-11-21 (Fri) 23:59. Summarize the GPT paper (A. Radford et al., "Improving Language Understanding by Generative Pre-Training," 2018) in about 4 pages (Word or PDF), in a free format.
- **Individual project proposal presentation:** Announced in Week 12 and held in Weeks 13-15 (11/25, 12/2, 12/9). Each student presented a proposal for an artificial intelligence application project that uses deep learning for about 5 minutes.
  - The topic was free, and the presentations were assessed on feasibility, technical understanding, creativity, and similar criteria.
  - Students were encouraged, but not required, to include the problem definition and solution, the dataset (how to obtain it for an open dataset or how to collect it otherwise), a specific deep learning model (e.g., YOLOv11 rather than just an object detection model), and how the output of the deep learning model would be used.
  - Each student prepared presentation slides (ppt), presented them, and submitted them to the assignment board afterwards.

---

<br>

## 4. Course Policies

- **Academic Integrity:** Students participating in the university's educational activities have the following general duties throughout the curriculum, including attending lectures, preparing and submitting assignments, and taking exams.
  1. Participate actively in all learning with a sincere and honest mind and an enthusiastic attitude, in accordance with the purpose of the university and its educational philosophy.
  2. Be aware of one's position as an educated person receiving higher education and devote oneself to creative academic inquiry based on moral sensitivity.
  3. Faithfully keep the promises made to faculty members or fellow students in connection with educational activities.
  4. Respect the dignity of faculty members and fellow students and observe basic courtesy.
- If credits are found to have been obtained through dishonest acts such as plagiarizing assignments, reusing assignments from other classes, or cheating on exams, or through other improper means, the credits are canceled even if already recognized, and disciplinary action such as reprimand, suspension, expulsion, or dismissal may be taken in consideration of the circumstances.
- To raise awareness of academic ethics and foster creativity, the university operates a plagiarism detection system (SafeAssign in Blackboard), and students must check in advance whether their assignments constitute plagiarism when submitting them.

---

<br>

## 5. Weekly Plan and Progress

| Week | Class Dates (Tue) | Planned Topic | Lecture Decks Covered |
|:-----|:-----|:-----|:-----|
| W01 | 9/2 | Introduction |  |
| W02 | 9/9 | Deep learning fundamentals 1 | 1_소프트웨어응용_딥러닝기초 (Deep Learning Fundamentals)<br>1-1_load_vgg.ipynb |
| W03 | 9/16 | Image processing basics | 2_소프트웨어응용_CNN_RNN (CNN and RNN)<br>2_CNNRNN.ipynb |
| W04 | 9/23 | Image processing applications 1 | 3_소프트웨어응용_영상처리1 (Image Processing 1)<br>3-1_training_tutorial.ipynb<br>3-2_ResNet_simple.ipynb |
| W05 | 9/30 | Image processing applications 2 | 4_소프트웨어응용_영상처리2 (Image Processing 2)<br>Assignment #1 (due 10/17) |
| W06 | 10/7 | Sound processing basics | No class (Chuseok holidays) |
| W07 | 10/14 | Sound processing applications 1 | 5_소프트웨어응용_영상처리3 (Image Processing 3) |
| W08 | 10/21 | Midterm Exam |  |
| W09 | 10/28 | Text processing basics | 6_소프트웨어응용_소리처리 (Sound Processing) |
| W10 | 11/4 | Text processing applications 1 | 7_소프트웨어응용_자연어처리1 (Natural Language Processing 1) |
| W11 | 11/11 | Text processing applications 2 | 8_소프트웨어응용_자연어처리2 (Natural Language Processing 2)<br>Assignment #2 (due 11/21) |
| W12 | 11/18 | Lightweight models | 9_소프트웨어응용_경량모델 (Lightweight Models)<br>Proposal presentation announcement |
| W13 | 11/25 | LLM applications | Individual project proposal presentations |
| W14 | 12/2 | Presentation sessions | Individual project proposal presentations |
| W15 | 12/9 | Presentation sessions | Individual project proposal presentations |
| W16 | 12/16 | Final Exam |  |

- The planned topics follow the weekly study contents of the syllabus. The syllabus plans Week 1 and Weeks 14-15 as lectures and discussion and Weeks 2-7 and 9-13 as lectures and practice, and it assigns creative convergence as the core competency of Weeks 14-15 and practical application as that of the other weeks.
- Because Week 6 was canceled for the Chuseok holidays, the actual progress fell behind the plan. Deep learning fundamentals were covered in Weeks 2-3, image processing in Weeks 4, 5, and 7, sound processing in Week 9, natural language processing in Weeks 10-11, and lightweight models in Week 12.
- The lecture decks map to the lecture notes as follows. The deck numbers equal the `Lec##` numbers.
  - Decks 1-9: [Lec01](Lecture-Notes/Lec01_Deep-Learning-Fundamentals/Concepts_Lecture.md) through [Lec09](Lecture-Notes/Lec09_Lightweight-Models/Concepts_Lecture.md), with the same numbers
  - `1-1_load_vgg.ipynb`: the [Lec01 lab notes](Lecture-Notes/Lec01_Deep-Learning-Fundamentals/Concepts_Lab.md) and the example notebook `Example_Load-VGG.ipynb`
  - `2_CNNRNN.ipynb`: the [Lec02 lab notes](Lecture-Notes/Lec02_Deep-Learning-Fundamentals-2/Concepts_Lab.md) and the example notebook `Example_CNN-RNN.ipynb`
  - `3-1_training_tutorial.ipynb`, `3-2_ResNet_simple.ipynb`: the [Lec03 lab notes](Lecture-Notes/Lec03_Image-Processing-1/Concepts_Lab.md) and the example notebooks `Example_Training-Tutorial.ipynb` and `Example_ResNet-Simple.ipynb`
- Page 36 of deck 4 (the Assignment 1 notice), page 42 of deck 8 (the Assignment 2 notice), and pages 30-31 of deck 9 (the proposal presentation announcement and reference sites) are course operations, so they are not included in the lecture notes. The assignments are summarized in [Section 3](#3-assessment).

---

<br>

## 6. References

| Type | Contents |
|:-----|:-----|
| Textbook | No designated textbook; lectures use the lecture slides distributed by the instructor |
| Lecture Slides | Nine lecture slide decks (pptx) and four Jupyter notebooks for practice distributed by the instructor |
| Designated Book | I. Oh and J. Lee, *IT CookBook, 파이썬으로 만드는 인공지능* (Artificial Intelligence with Python). Seoul, South Korea: Hanbit Academy, 2021. |
| Designated Book | Y. Ogawa, *만들면서 배우는 파이토치 딥러닝*, 박광수, Trans. Seoul, South Korea: Hanbit Media, 2021. The original is Y. Ogawa, *つくりながら学ぶ! PyTorchによる発展ディープラーニング* (Mynavi Publishing, 2019). The Week 2 practice notebook (`1-1_load_vgg.ipynb`) follows the VGG example in Chapter 1 of this book. |

- The two books are listed in the syllabus as designated books and reference materials, and the order of the lecture slides does not follow the chapter structure of either book.
