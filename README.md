# 2024 월간 데이콘 저해상도 조류 이미지 분류 AI 경진 대회

public, private 리더보드에서 높은 점수를 기록하였습니다.

- publice LB score : 0.98335 ( 1위 )
- private LB score : 0.98498 ( 1위 )

## 1. 환경 설정
### 1.1 저장소 코드 복사

```bash
git clone https://github.com/ujinyang/dacon2024_bird_lowres_image_classification.git
cd dacon2024_bird_lowres_image_classification
```

### 1.2 패키지 설치

wand (ImageMagicK) 의 경우, 추가 라이브러리를 설치하여야 합니다.

```bash
sudo apt install -y libmagickwand-dev
pip install -r requirements.txt
```

## 2. 폴더 구조

```bash
├── train
│   ├── TRAIN_00000.jpg
│   ├── TRAIN_00001.jpg
│   ├── ...
│   └── TRAIN_xxxxx.jpg
├── test
│   ├── TEST_00000.jpg
│   ├── TEST_00001.jpg
│   ├── ...
│   └── TEST_xxxxx.jpg
├── sample_submission.csv
├── train.csv
├── test.csv
├── basslibrary_model_train.ipynb
├── basslibrary_model_submit.ipynb
└── ckpt
```

## 3. 모델 훈련

※ 노트북 파일에 실행결과가 포함되어 있지 않은 대신, 별도의 출력로그를 첨부하였습니다.

### 3.1 eva_large 모델

basslibrary_model_train.ipynb 파일에서 CFG 설정을 아래처럼 변경하여 훈련
EMA(Exponential Moving Average) 모델을 포함하여 모두 10개의 체크포인트 파일 생성

```bash
CFG['MODEL_NAME'] = "timm/eva_large_patch14_196.in22k_ft_in22k_in1k"
CFG['IMG_SIZE'] = 196
CFG['BATCH_SIZE'] = 48
CFG['LR'] = [ 0.25e-5 * np.sqrt(CFG['BATCH_SIZE']), 1e-6 ]
```

[출력로그](https://github.com/ujinyang/model_run(eva_large).log)

### 3.2 beitv2_large 모델

basslibrary_model_train.ipynb 파일에서 CFG 설정을 아래처럼 변경하여 훈련
EMA(Exponential Moving Average) 모델을 포함하여 모두 10개의 체크포인트 파일 생성

```bash
CFG['MODEL_NAME'] = "timm/beitv2_large_patch16_224.in1k_ft_in22k_in1k"
CFG['IMG_SIZE'] = 224
CFG['BATCH_SIZE'] = 48
CFG['LR'] = [ 0.25e-5 * np.sqrt(CFG['BATCH_SIZE']), 1e-6 ]
```

[출력로그](https://github.com/ujinyang/model_run(beitv2_large).log)

## 4. 모델 결과 제출

※ 노트북 파일에 실행결과가 포함되어 있지 않은 대신, 별도의 출력로그를 첨부하였습니다.

### 4.1 모델 앙상블
basslibrary_model_submit.ipynb 파일을 실행하면, ckpt폴더에 저장된 결과 중
EMA를 제외한 일반 모델 10개를 기준으로 submit 파일을 생성함

|No.| model_name                                  |fold|epoch|val_loss|val_score|
|---|---------------------------------------------|----|-----|--------|---------|
| 1 | eva_large_patch14_196.in22k_ft_in22k_in1k   |  0 |   5 | 0.4111 |  0.9811 |
| 2 | eva_large_patch14_196.in22k_ft_in22k_in1k   |  1 |  11 | 0.4039 |  0.9825 |
| 3 | eva_large_patch14_196.in22k_ft_in22k_in1k   |  2 |   8 | 0.4083 |  0.9811 |
| 4 | eva_large_patch14_196.in22k_ft_in22k_in1k   |  3 |   5 | 0.4032 |  0.9819 |
| 5 | eva_large_patch14_196.in22k_ft_in22k_in1k   |  4 |   6 | 0.4065 |  0.9837 |
| 6 | beitv2_large_patch16_224.in1k_ft_in22k_in1k |  0 |   9 | 0.4112 |  0.9818 |
| 7 | beitv2_large_patch16_224.in1k_ft_in22k_in1k |  1 |  10 | 0.4071 |  0.9815 |
| 8 | beitv2_large_patch16_224.in1k_ft_in22k_in1k |  2 |   9 | 0.4129 |  0.9803 |
| 9 | beitv2_large_patch16_224.in1k_ft_in22k_in1k |  3 |  10 | 0.4049 |  0.9825 |
|10 | beitv2_large_patch16_224.in1k_ft_in22k_in1k |  4 |   6 | 0.4086 |  0.9813 |


[출력로그](https://github.com/ujinyang/model_result(ensemble).log)
