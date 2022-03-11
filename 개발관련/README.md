# AI 개발관련 지식
## 하이퍼 파라미터는 무엇인가?
하이퍼 파라미터는 모델의 학습에 필요한 수동 설정값이다.

## Non-Linearity라는 말의 의미와 그 필요성은?
Non-Linearity는 비 선형성이라는 뜻이다. 데이터의 복잡도가 높아지고 차원이 높아지게 되면 데이터의 분포는 단순한 선형이 아닌 비 선형 형태를 가지기 때문에 단순한 1차식의 형태로는 데이터를 표현할 수 없기에 Non-Linearity가 중요하다.

## Sigmoid보다 ReLU를 많이 쓰는 이유는?
Sigmoid의 문제점으로는 입력 값이 일정 범위를 넘어가게 되면 0또는 1로 수렴해버리고 gradient가 0으로 수렴해 버리게 되어 학습이 제대로 되지 않는다. 또한 Sigmoid는 범위가 0에서 1이기 때문에 중앙값이 0이 아니게 된다. 0을 기준으로 데이터가 분포하게 되었을 때가 이상적인데 Sigmoid는 그것을 만족시키지 못한다. 또한 ReLU보다 연산에 많은 cost가 소모된다.

## ReLU는 그래프만 보면 곡선이 아닌 것 같은데 어떻게 Non-Linearity인가?
ReLU를 보면 단순 Linear 형태인 듯 보이지만, Multi-layer의 activation function으로 ReLU를 사용하게 되면 Linear 한 부분 부분의 결합의 합성 함수가 만들어 지게 되는데, 이 결합 구간을 보았을 때 최종적으로 Non-Linearity한 성질을 가지게 된다.

## ReLU의 문제점은?
ReLU의 문제점은 원래 ReLU가 max(0, x)이기 때문에 0보다 작으면 함수 미분값이 0이된다는 약점이 있다. 이 문제를 해결하기 위해 LeakyReLU라는 함수가 만들어졌다.

## Bias는 뭘까?
Bias는 모델이 데이터에 잘 fitting하게 하기 위하여 평행 이동하는 역할을 한다. 데이터를 2차원으로 표현했을 때, 모든 데이터가 원점기준에 분포해 있지는 않기 때문에 Bias를 이용하여 모델이 평면 상에서 이동할 수 있도록 하고, 이 Bias또한 학습하게 한다.

## Gradient Descent란?
Gradient Descent, 즉 경사 하강 알고리즘은 Cost Function(비용 함수)의 값을 최소화하는 파라미터를 찾는 알고리즘이다. 기본적인 개념은 함수의 기울기를 구하여 기울기가 낮은 쪽으로 계속 이동시켜 최적값에 이를 때까지 반복하는 것이다. Gradient Descent의 약점은 현재 위치에서의 기울기를 사용하기 때문에 local minimum에 빠질 수 있다는 점이다. 그래서 추후에 모멘텀이라는 방식이 등장하게 된다.

## Gradient Descent를 써야하는 이유는?
gradient를 통해 계산된 loss를 줄이기 위해, 역전파를 토대로 파라미터 값을 업데이트 하기 때문에 GD는 필수적으로 써야한다.

## Gradient Descent 종류에 대한 각각을 설명한다면?
![참조](https://www.dropbox.com/s/k03ffo5rjlwc03z/optimizers.png?raw=1)

## Loss Surface란?
모델을 훈련시킨다는 말은 비용 함수에서 파라미터를 업데이트하며 global minimum을 찾는 과정이다. Loss Surface란 global minimum을 찾아가는 과정을 시각화한 것으로, 모델을 이해하고 설계하는데 인사이트를 준다.

## Dropout이란?
Dropout은 노드들의 연결을 무작위로 끊는 방식으로, 하나의 노드가 너무 큰 가중치를 가져 다른 노드들의 학습을 방해하는 현상을 억제한다. 이를 통해 모델의 일반화 성능을 높이고 Overfitting을 방지하는 효과를 얻을 수 있다. 

## Average Pooling과 Max Pooling의 차이점은?
Pooling은 Feature map에서 feature 수를 감소시키는 역할을 한다. Average Pooling은 kernel window에 해당하는 값들의 평균을 대표로, Max Pooling은 가장 큰 값을 대표로 feature 수를 감소시킨다. Max pooling은 kernel 영역 내에서 가장 두드러지는 값을 남기고, average는 영역 내의 모든 값을 고려하는 효과가 있다.

## Localization이란?
Object detection에서 검출한 객체의 위치 정보를 예측하는 것을 의미한다. 네트워크의 output vector에 좌표 정보를 출력하게끔 학습할 수 있다.

## Attention이란?
RNN에 기반한 모델이 가지고 있는 문제는 기울기 소실(Vanishing Gradient)과 정보 손실 문제이다. 이로 인해 input으로 들어가는 문장이 길어지면 품질이 떨어지는 현상이 나타난다. 이를 해결하기 위해 등장한 방법이 바로 Attention이다. Attention mechanism은, 디코더에서 출력 단어를 예측하는 매 시점마다 인코더에서의 전체 입력 문장 중 해당 시점에서 예측해야 할 단어와 연관성있는 부분을 좀 더 집중해서 보는 것이다.

## Transformer란?
Transformer는 2017년 구글이 발표한 논문인 "Attention is all you need"에서 나온 모델로 seq2seq의 구조인 인코더-디코더를 따르면서도, 논문의 이름처럼 Attention으로만 구현한 모델이다. Transformer는 RNN을 사용하지 않고, 인코더-디코더 구조를 설계하였음에도 번역 성능에서 RNN을 능가하는 성능을 보여주었다.

## Collaborative filtering이란?
Collaborative filtering은 협업 필터링이라 하여 추천 시스템에서 가장 많이 사용되는 기술로, 유저-아이템 간 상호 작용 데이터를 활용하는 방법론이다. 예를 들면 "이 영화를 좋아했던 다른 사람들은 또 어떤 영화를 좋아해요?"

협업 필터링 뿐만 아니라 추천 모델에는 콘텐츠 기반 필터링(Content-based Filtering)이 있는데, [kakao Tech](https://tech.kakao.com/2021/10/18/collaborative-filtering/)에서 진짜 야무지게 설명해놓았다. 참고하면 좋다.

## Few-Shot Learning이란?
Few-Shot Learning은 쉽게 말하자면 훈련을 통해 class를 분류하는 것을 배우고, image가 들어왔을 때 어떤 것과 같은 class인지 구분하는 일을 하도록 하는 것이다. input image가 어떤 class에 속하는지를 배우는게 아닌, 어떤 class와 같은 class이냐를 배우는 것이다. 

## Federated Learning이란?
Federated Learning이란 "연합 학습"으로써 다수의 로컬 클라이언트와 하나의 중앙 서버가 협력하여 데이터가 탈중앙화된 상황에서 글로벌 모델을 학습하는 기술이다. 여기서 로컬 클라이언트는 사물 인터넷 기기, 스마트 폰 등을 말한다. Federated Learning은 데이터 프라이버시 향상과 커뮤니케이션 효율성, 이 두 가지 장점때문에 굉장히 유용하다.

## 딥러닝을 할 때 GPU를 쓰는 이유는?
GPU에는 부동소수점 계산에 특화된 수많은 코어가 있어서 병렬 처리를 수행하기에 유리하기 때문이다.