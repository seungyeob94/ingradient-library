# ingradient library
 
## 0. 다운로드
pip install ingradient-library-temp

## 1. 이미지 포멧 맞추기
![스크린샷 2021-09-23 오후 4 36 04](https://user-images.githubusercontent.com/87344797/134470205-83603804-7556-402c-833a-1b919b7a16db.png)
- 위와 같은 구성을 따른다.
- 인덱싱에 딱히 기준은 없으나 파일 목록을 npz파일과 pkl파일로 나눈 후, sorting해서 dataset에서 가져오므로 npz파일과 pkl파일은 같은 순번을 따르도록 한다.

![스크린샷 2021-09-23 오후 4 41 29](https://user-images.githubusercontent.com/87344797/134470839-ee7ccc7b-7182-43ac-9425-2f83daa59d1a.png)
- npz 파일의 dimension은 [modalities, z, x, y] 와 같이 이루어지며, dataset을 로딩하는 과정에서 direction을 보정해주기 때문에 단순히 위와 같이 저장하면 된다.
- 저장 전에 각 데이터에 대해 Non-zero Cropping이 들어가며 이는 ingradient_library.preprocessing의 Cropping Method를 사용하면 된다.
- 해당 예시는 한 데이터에 대한 저장 방법이고 이를 반복문을 사용해 활용한다.


## 2. Resampling
![스크린샷 2021-09-23 오후 4 58 09](https://user-images.githubusercontent.com/87344797/134472713-eef1a815-a090-4575-b30b-cf28e726e332.png)

- Get_target_spacing 객체를 만든다.
- 이 때 Anisotropy Threshold 값을 고를 수 있다. (2D는 아직 디버깅 진행하지 않음.) 또한 Default 값은 isotropy_percentile_value = 0.50, anisotropy_percentile_value = 0.90에 맞춰져 있다. 이를 변경하는 것도 가능하다.
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
- CustomDataset은 torch.utils.data에서 dataset 클래스를 상속 받았다.
- 데이터 셋이 저장된 디렉토리의 PATH를 입력한다.
- 이전에 선언한 normalizer를 가져온다.

## 5. DataLoader
![스크린샷 2021-09-23 오후 5 05 22](https://user-images.githubusercontent.com/87344797/134473519-9f42480a-226d-49da-bcd6-28bce29c4bcf.png)



