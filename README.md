# ingradient library
 
## 0. 다운로드
pip install ingradient-library-temp

## 1. 이미지 포멧 맞추기
![스크린샷 2021-09-23 오후 4 36 04](https://user-images.githubusercontent.com/87344797/134470205-83603804-7556-402c-833a-1b919b7a16db.png)
> 위와 같은 구성을 따른다.
> 인덱싱에 딱히 기준은 없으나 파일 목록을 npz파일과 pkl파일로 나눈 후, sorting해서 dataset에서 가져오므로 npz파일과 pkl파일은 같은 순번을 따르도록 한다.

![스크린샷 2021-09-23 오후 4 41 29](https://user-images.githubusercontent.com/87344797/134470839-ee7ccc7b-7182-43ac-9425-2f83daa59d1a.png)
> npz 파일의 dimension은 [modalities, z, x, y] 와 같이 이루어지며, dataset을 로딩하는 과정에서 direction을 보정해주기 때문에 단순히 위와 같이 저장하면 된다.
> 저장 전에 각 데이터에 대해 Non-zero Cropping이 들어가며 이는 ingradient_library.preprocessing의 Cropping Method를 사용하면 된다.
> 해당 예시는 한 데이터에 대한 저장 방법이고 이를 반복문을 사용해 활용한다.


## 2. Resampling
![스크린샷 2021-09-23 오후 4 58 09](https://user-images.githubusercontent.com/87344797/134472713-eef1a815-a090-4575-b30b-cf28e726e332.png)

> Get_target_spacing 객체를 만든다.
> 이 때 Anisotropy Threshold 값을 고를 수 있다. (2D는 아직 디버깅 진행하지 않음.) 또한 Default 값은 isotropy_percentile_value = 0.50, anisotropy_percentile_value = 0.90에 맞춰져 있다. 이를 변경하는 것도 가능하다.
> run 메소드를 사용해 Dataset들이 저장된 폴더로부터 Spacing 값들을 구한다. 이 떄 각 spacing 값들이 해당 객체안에 저장된다.
> Target Spacing값과 Anisotropy axis에 해당하는 index를 얻는다.
> 해당 값들을 Resampling 객체를 만들 때 넣어준다.
> 뒤에 이 Resampling 객체는 DataLoader에 들어가게 되며, 이 후 자동으로 patch를 뽑을 때 마다 Resampling을 진행한다.
