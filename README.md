# ingradient library
 
## 0. 다운로드
pip install ingradient-library-temp

## 1. 이미지 포멧 맞추기
![스크린샷 2021-09-23 오후 4 36 04](https://user-images.githubusercontent.com/87344797/134470205-83603804-7556-402c-833a-1b919b7a16db.png)
- 위와 같은 구성을 따른다.
- 인덱싱에 딱히 기준은 없으나 데이터 파일들을 npz파일과 pkl파일로 나눈 후, sorting해서 순서대로 dataset에서 가져오므로 npz파일과 pkl파일은 같은 순번을 따르도록 한다.

![스크린샷 2021-09-23 오후 4 41 29](https://user-images.githubusercontent.com/87344797/134470839-ee7ccc7b-7182-43ac-9425-2f83daa59d1a.png)
- npz 파일의 dimension은 [modalities, z, x, y] 와 같이 이루어지며, dataset을 로딩하는 과정에서 direction을 보정해주기 때문에 단순히 위와 같이 저장하면 된다.
- 저장 전에 각 데이터에 대해 Non-zero Cropping이 들어가며 이는 ingradient_library.preprocessing의 Cropping Method를 사용하면 된다.
- 해당 예시는 데이터 한 개에 대한 저장 방법이고 이를 반복문을 사용해 활용한다.


## 2. Resampling
![스크린샷 2021-09-23 오후 4 58 09](https://user-images.githubusercontent.com/87344797/134472713-eef1a815-a090-4575-b30b-cf28e726e332.png)

- Get_target_spacing 객체를 만든다.
- 이 때 Anisotropy Threshold 값을 고를 수 있다 Default 값은 3이다. 이 외에도 isotropy_percentile_value = 0.50, anisotropy_percentile_value = 0.90에 Default가 맞춰져 있다. 이를 변경하는 것도 가능하다.
- run 메소드를 사용해 Dataset들이 저장된 폴더로부터 Spacing 값들을 구한다. 이 떄 각 spacing 값들이 해당 객체안에 저장된다.
- Target Spacing값과 Anisotropy axis에 해당하는 index를 얻는다.
- 해당 값들을 Resampling 객체를 만들 때 넣어준다.
- 뒤에 이 Resampling 객체는 DataLoader에 들어가게 되며, 이 후 자동으로 patch 단위로 Resampling을 진행한다.


## 3. Normalizer
![스크린샷 2021-09-23 오후 5 00 30](https://user-images.githubusercontent.com/87344797/134472958-d024dce0-c524-4fb9-9e4d-24c9f8a9d30a.png)

- Normalizer를 생성한다. 생성 시에 percentile clipping 값을 설정할 수 있다.
- nnUNet의 기본 세팅은 MRI의 경우 percentile clipping을 사용하지 않고, CT의 경우 [0.05, 0.95]를 따른다.
- 이 후, dataset 메소드에서 normalizer를 가져와 데이터를 샘플링 할 때 마다 한 patient 단위로 normalization을 수행한다.

## 4. Data Augmentation
![스크린샷 2021-09-23 오후 5 05 22](https://user-images.githubusercontent.com/87344797/134473519-9f42480a-226d-49da-bcd6-28bce29c4bcf.png)

- nnUNet에 들어가는 데이터 어그멘테이션 메소드를 가져온다.
- 만약 해당 메소드를 사용하고 싶지 않다면 prob 값을 0으로 하면 된다.
- 구성을 향후 수정해 더 많은 Data Augmentation을 가져올 예정.
- 각 데이터 어그멘테이션 별 메소드는 from ingradient_library import patch_transform 상에서 확인할 수 있다.  

## 5. Dataset
![스크린샷 2021-09-23 오후 5 03 15](https://user-images.githubusercontent.com/87344797/134473278-93b2df7a-04ee-411b-a7f8-592e90d05fdb.png)
- CustomDataset은 torch.utils.data에서 dataset 클래스를 상속 받았다. 때문에 pytorch의 random split 연산을 사용할 수 있다.
- 데이터 셋이 저장된 디렉토리의 PATH를 입력한다.
- 이전에 선언한 normalizer를 가져온다.

## 5. DataLoader
![스크린샷 2021-09-23 오후 5 08 04](https://user-images.githubusercontent.com/87344797/134473860-3686453b-9d24-43dc-9d54-aa8b1a2d109e.png)
- Batch Size는 하나의 Patient에서 뽑아내게 되며 기본적으로 nnUNet이 사용한 Oversampling이 지원된다.
- iteration은 각 patient별로 몇번의 샘플링을 진행할 지 정한다. 만약 batch size = 2, iteration = 2 라면 1epoch 마다 하나의 patient에서 4번의 샘플링을 진행한다.
- Data Augmentation 객체를 넣어서 Patch 단위로 Augmentation을 진행한다.
- Resampling 객체를 사용해 Patch 단위로 Resampling을 진행한다. 이는 GPU 연산의 효율성과 계산 복잡도 증가를 위해 뒤에서 사용했다.


## 6. Deep Supervision Model
![스크린샷 2021-09-23 오후 5 11 48](https://user-images.githubusercontent.co![Uploading 스크린샷 2021-09-23 오후 5.12.43.png…]()
m/87344797/134474332-bfa1c66c-49c0-4009-876b-ec335b19d359.png)

![스크린샷 2021-09-23 오후 5 12 56](https://user-images.githubusercontent.com/87344797/134474469-5feaca9b-45d6-4a2d-be3e-b68d7dadc14b.png)

- nnUNet에서 사용한 Deep supervision model을 기본적으로 지원한다.
- Loss 역시 연산 식이 기존과 다르므로 지원한다.

## 7. Training Example
![스크린샷 2021-09-23 오후 5 13 36](https://user-images.githubusercontent.com/87344797/134474568-c18322e1-50b8-4ce8-aaf2-719f772519c3.png)
- 다음과 같은 구성을 따른다. nnUNet의 learning rate decay를 명시해주는 것이 필요하다.
- Trainer를 구현하는 작업도 진행할 계획이다.

## 8. Inference
![스크린샷 2021-09-23 오후 5 14 30](https://user-images.githubusercontent.com/87344797/134474697-2e4da734-4704-4724-ab84-4009f820c0f0.png)

- dataset에 대해 Inference를 진행한다. dataset은 CustomDataset에 해당하는 모듈이다.
- mode = 'save' 일 경우 결과물을 save_path에 저장한다. mode = 'dice' 일 경우 각 파일들의 dice score를 print 한다.


## 9. Visualization
![스크린샷 2021-09-23 오후 5 16 42](https://user-images.githubusercontent.com/87344797/134475015-244de41f-b097-4eba-b3fb-5ff2469df5d1.png)
- Training 시에 Segmentation Output을 각 Deepsupervision Layer 별로 지원한다.
- 이 외에도 기본 plain unet에 대한 시각화도 지원한다.





