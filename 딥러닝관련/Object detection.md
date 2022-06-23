## Ground truth란?

딥러닝 관점에서 Ground truth란 학습하고자 하는 데이터의 원본 혹은 실제 값을 표현할 떄 사용된다.

## Occlusion이란?

Object tracking 분야에서 물체가 겹치거나 가려 인식을 못하는 경우를 말한다.

## NMS란?

NMS는 Non-maximum suppression의 약자로 Object detection분야에서 주어진 bbox의 집합이 주어졌을 때, 최종적으로 남길 bbox를 담을 리스트를 생성하고 전체 집합에서 삭제하고 최종 리스트에 추가하고 class score가 너무 낮은 bbox또한 전체 집합에서 삭제하는 역할을 한다. 또한 최종 리스트에서 가장 class sore가 높은 bbox와 전체 집합에 담긴 bbox의 IOU를 계산하여 주어진 임계치보다 크다면 B에서 제거하는 역할을 한다.

## Precision이란?

Precision은 모든 검출 결과 중 옳게 검출한 비율을 의미한다. 

![](https://velog.velcdn.com/images/sanha9999/post/e2b44d90-db62-4019-9bb7-f52535ef5a14/image.png)

TP : True Positive = 검출한 결과가 옳은 것 = 기계가 맞다고 한게 맞은것

FP : False Positive = 검출한 결과가 틀린 것 = 기계가 맞다고 한게 틀린 것

## Recall이란?

![](https://velog.velcdn.com/images/sanha9999/post/820e3f00-f886-4f3c-a1e0-4ad9eba16256/image.png)

FN : False Negative = 검출되었어야 하는 물체인데 검출되지 않은 것 = 기계가 못찾은것

## Confusion Matrix
![](https://velog.velcdn.com/images/sanha9999/post/87747789-6047-4f5a-9bfd-519bec2d2a18/image.png)

## mAP
![](https://velog.velcdn.com/images/sanha9999/post/56807d40-aa67-45c0-b83e-54f77d2dc9ae/image.png)

인식 알고리즘의 성능을 하나의 값으로 표현한 것으로 precision-recall 곡선에서 그래프 선 아래 쪽의 면적으로 계산 된다. Average Precision이 높으면 높을 수록 그 알고리즘의 성능이 전체적으로 우수하다는걸 의미한다. 컴퓨터 비전 분야에서 물체인식 알고리즘의 성능은 대부분 Average Precision으로 평가한다.