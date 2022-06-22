## Ground truth란?

딥러닝 관점에서 Ground truth란 학습하고자 하는 데이터의 원본 혹은 실제 값을 표현할 떄 사용된다.

## Occlusion이란?

Object tracking 분야에서 물체가 겹치거나 가려 인식을 못하는 경우를 말한다.

## NMS란?*

NMS는 Non-maximum suppression의 약자로 Object detection분야에서 주어진 bbox의 집합이 주어졌을 때, 최종적으로 남길 bbox를 담을 리스트를 생성하고 전체 집합에서 삭제하고 최종 리스트에 추가하고 class score가 너무 낮은 bbox또한 전체 집합에서 삭제하는 역할을 한다. 또한 최종 리스트에서 가장 class sore가 높은 bbox와 전체 집합에 담긴 bbox의 IOU를 계산하여 주어진 임계치보다 크다면 B에서 제거하는 역할을 한다.