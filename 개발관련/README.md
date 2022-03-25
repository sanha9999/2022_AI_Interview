# AI 개발관련 지식
## 텐서플로와 파이토치의 차이점은 무엇인가?
![참조](https://cdn.aitimes.com/news/photo/202010/132756_129974_5632.png)

## 머신러닝이란 무엇인가?
학습을 통해 자동으로 개선하는 알고리즘으로, 인간의 사고를 효율적인 계산의 관점에서 보고 모방하는 알고리즘을 말한다.

## Supervised와 Unsupervised의 차이는 무엇일까?
Surpervised Learning은 지도 학습으로 정답을 알려주며 학습하는 것이다. label이 있는 data가 필요하다. Unsupervised Learning은 비지도 학습으로 정답을 따로 알려주지 않고 학습하는 것이다. 대표적으로 지도학습은 분류 + 회귀 문제에 많이 사용되고 비지도 학습은 군집화 문제에 사용된다. 

## 강화학습이란 무엇인가?
상/벌을 주어 상을 최대화하고 벌을 최소화하는 값을 찾는 것을 학습하는 것이다. 대표적으로 그 유명한 알파고가 강화학습으로 학습되었다.

## 딥러닝은 무엇이고 머신러닝과 차이는 무엇인가?
딥러닝은 인공신경망을 깊은 구조로 설계하여 비선형성이 높은 문제들을 해결하는 방법을 통칭한다. 딥러닝은 머신러닝의 일종이며 모델 내부의 결정 과정을 해석하기 불가능한 **블랙박스**구조로 되어있다. 보통, 딥러닝이 일반적인 머신러닝 기법보다 더 많은 데이터를 필요로하며 계산량이 압도적으로 많다.

## 딥러닝에서 '표현을 학습하다'의 의미는 무엇인가?
머신러닝/딥러닝의 표현이란 데이터를 인코딩하거나 묘사하기 위해 데이터를 바라보는 다른 방법을 말한다.

## Cost Function과 Activation Function은 무엇인가?
Cost Function은 모델이 도출한 결과(output)와 목표 결과(target)를 어떻게 비교할 것인가를 의미한다. 두 결과의 차이를 의미하는 Cost는 Optimizer에 의해 Parameter가 갱신될 때, Step size를 얼마나 크게 가져갈 것인가에 결정적인 영향을 끼친다. Activation Function은 뉴런이 유입되는 신호로부터 도출하는 값을 정제하는 역할을 한다. Activation Function의 종류에는 sigmoid, Relu, Hyperbolic Tangent 등이 있다. 

## Local Minima와 Global Minima는 무엇인가?
Global minima는 Gradient Descent 방법으로 학습시에, 해당 도메인에 대해 가장 낮은 cost를 가질 수 있는 weight가 위치한 지점이다. Local Minima는 Gradient Descent로 Global Minima를 찾아가는 과정에서 주변에 지역적으로 Gradient 하강, 상승 구간이 존재하여 빠질 수 있는 가짜 Minima이다.

## 차원의 저주란 무엇인가?
차원의 저주는 한 샘플을 특정짓기 위해 많은 정보(다양한 차원의)를 수집할수록 오히려 데이터 사이의 거리가 멀어져 차원에 빈공간이 생기기 때문에 학습이 어려워지는 문제이다. 해결방법은 데이터의 밀도가 높아질때까지 데이터를 모아 훈련 세트를 키우거나, PCA같은 차원 축소 기법을 이용하여 해결한다.

## PCA기법은 무엇인가?
고차원의 데이터를 저차원의 데이터로 축소시키는 차원 축소 방법중 하나로, 훈련 데이터의 많은 feature중 중요한 feature 몇개만 뽑아내는 방법이 PCA이다. [블로그](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-9-PCA-Principal-Components-Analysis)에서 잘 정리되어있다.

## 하이퍼 파라미터는 무엇인가?
하이퍼 파라미터는 모델의 학습에 필요한 수동 설정값이다.

## Non-Linearity라는 말의 의미와 그 필요성은?
Non-Linearity는 비 선형성이라는 뜻이다. 데이터의 복잡도가 높아지고 차원이 높아지게 되면 데이터의 분포는 단순한 선형이 아닌 비 선형 형태를 가지기 때문에 단순한 1차식의 형태로는 데이터를 표현할 수 없기에 Non-Linearity가 중요하다.

## 활성화 함수(Activation Function)의 종류
* Sigmoid : 역전파 과정에서 기울기가 소실되는 문제 발생
* Tanh : Sigmoid의 기울기 소실 문제를 해결함
* ReLU : 음수일때 기울기가 0이 되버림
* Leaky ReLU : 음수에서 기울기가 0이 되는 ReLU 문제 해결
* Softmax : 다중 클래스 분류 시 사용함

## Sigmoid보다 ReLU를 많이 쓰는 이유는?
Sigmoid의 문제점으로는 입력 값이 일정 범위를 넘어가게 되면 0또는 1로 수렴해버리고 gradient가 0으로 수렴해 버리게 되어 학습이 제대로 되지 않는다. 또한 Sigmoid는 범위가 0에서 1이기 때문에 중앙값이 0이 아니게 된다. 0을 기준으로 데이터가 분포하게 되었을 때가 이상적인데 Sigmoid는 그것을 만족시키지 못한다. 또한 ReLU보다 연산에 많은 cost가 소모된다.

## ReLU는 그래프만 보면 곡선이 아닌 것 같은데 어떻게 Non-Linearity인가?
ReLU를 보면 단순 Linear 형태인 듯 보이지만, Multi-layer의 activation function으로 ReLU를 사용하게 되면 Linear 한 부분 부분의 결합의 합성 함수가 만들어 지게 되는데, 이 결합 구간을 보았을 때 최종적으로 Non-Linearity한 성질을 가지게 된다.

## ReLU의 문제점은?
ReLU의 문제점은 원래 ReLU가 max(0, x)이기 때문에 0보다 작으면 함수 미분값이 0이된다는 약점이 있다. 이 문제를 해결하기 위해 LeakyReLU라는 함수가 만들어졌다.

## Bias는 뭘까?
Bias는 모델이 데이터에 잘 fitting하게 하기 위하여 평행 이동하는 역할을 한다. 데이터를 2차원으로 표현했을 때, 모든 데이터가 원점기준에 분포해 있지는 않기 때문에 Bias를 이용하여 모델이 평면 상에서 이동할 수 있도록 하고, 이 Bias또한 학습하게 한다.

## Variance란 무엇일까?
머신러닝에서 추정값들의 흩어진 정도를 Variance라고 한다.

## Back Propagation이란?
신경망의 최종 단계에서 계산된 오차를 바탕으로 이전 단계의 파라미터를 업데이트하는 방법이다.

## Batch Normalization이란?
Batch Normalization, 즉 배치 정규화는 평균과 분산을 조정하는 과정이 별도의 과정으로 떼어진 것이 아닌 신경망 안에 포함되어 학습시 평균과 분산을 조정한다. 즉 각 레이어마다 정규화하는 레이어를 두는 것이 배치 정규화이다.

## Data Normalization은 무엇인가?
Data Normalization은 입력 데이터의 최소, 최대값을 일정 범위(0~1) 내로 조절하는 것이다. 이를 통해 특정 데이터가 결과에 대해 과도한 영향을 미칠 수 있는 지위를 획득하는 것을 방지할 수 있고, 모델의 학습을 원할하게 만든다.

## Weight Initialization이란 무엇인가?
Weight Initialization은 가중치 초기화라는 뜻으로, 딥러닝 학습에 있어 초기 가중치 설정은 매우 종요한 역할을 한다. 가중치를 잘못 설정할 경우 기울기 소실 문제나 표현력의 한계를 갖는 등 여러 문제를 야기할 수 있기 때문이다. Weight Initialization을 통해 Local minimum에 빠지는 문제를 해결할 수 있다.

## Overfitting(과적합) 피하기
* 모델을 간단하게 만들기 (파라미터 수 줄이기)
* k-fold cross validation
* 정규화를 사용한다.

## Gradient Descent란?
Gradient Descent, 즉 경사 하강 알고리즘은 Cost Function(비용 함수)의 값을 최소화하는 파라미터를 찾는 알고리즘이다. 기본적인 개념은 함수의 기울기를 구하여 기울기가 낮은 쪽으로 계속 이동시켜 최적값에 이를 때까지 반복하는 것이다. Gradient Descent의 약점은 현재 위치에서의 기울기를 사용하기 때문에 local minimum에 빠질 수 있다는 점이다. 그래서 추후에 모멘텀이라는 방식이 등장하게 된다.

## Gradient Descent를 써야하는 이유는?
gradient를 통해 계산된 loss를 줄이기 위해, 역전파를 토대로 파라미터 값을 업데이트 하기 때문에 GD는 필수적으로 써야한다.

## Gradient Descent 종류에 대한 각각을 설명한다면?
![참조](https://www.dropbox.com/s/k03ffo5rjlwc03z/optimizers.png?raw=1)

## Regularization이란 무엇인가?
단순히 Cost function의 값이 작아지는 방향으로 모델을 학습하면 특정 가중치가 과도하게 커지는 현상이 나타날 수 있기 때문에 Cost function을 계산할 때 가중치의 절대값 만큼을 더해주는 방법이다. Regularization을 통해 특정 가중치의 값이 너무 커지는 현상을 억제한다.

## 딥러닝을 할 때 GPU를 쓰는 이유는?
GPU에는 부동소수점 계산에 특화된 수많은 코어가 있어서 병렬 처리를 수행하기에 유리하기 때문이다.

## 학습 중인데 GPU를 100% 사용하지 않고 있다. 이유는 무엇일까?
GPU를 100% 활용하지 못하는 경우는 두가지로 나뉘는데, 첫번째로는 GPU 메모리를 충분히 활용하지 못하는 경우이고 두분째는 GPU의 연산 능력을 충분히 활용하지 못하는 경우이다. 메모리가 100%가 아닌 경우, 작은 Batch size가 문제일 수 있다. 연산 능력이 100%가 아닌 경우에는 모델의 계산 과정에서 CPU 병목이 원인일 수 있다.

## Batch size와 GPU 메모리의 관계는?
먼저 GPU 메모리란 GPU를 사용할 떄 한번에 가질 수 있는 메모리의 양이다. Batch size가 커지면 메모리의 양도 커진다. 그래서 필요한 메모리 양보다 GPU 메모리가 적을 경우에는 'out of memory'에러를 발생시키게 된다.

## Data parallelism이란?
Data parallelism은 학습해야할 데이터가 많은 상황에서 학습 속도를 높이기 위해 나온 분산 학습 방법이다. 예를 들자면 하나의 GPU가 1개의 Data를 학습해야 하는데 1분이 걸린다고 가정하면 1000개의 학습 데이터 Batch를 학습하려면 1000분이 걸리겠지만, 데이터를 100개씩 나눠 10개의 GPU에서 학습하면 100분이 걸릴 것이다. Data parallelism은 GPU의 수를 늘림을 통해 학습 데이터를 빨리 처리하여 시간을 단축하기 위한 전략이다.

## Loss Surface란?
모델을 훈련시킨다는 말은 비용 함수에서 파라미터를 업데이트하며 global minimum을 찾는 과정이다. Loss Surface란 global minimum을 찾아가는 과정을 시각화한 것으로, 모델을 이해하고 설계하는데 인사이트를 준다.

## Training 세트와 Test 세트를 분리하는 이유는?
실제 데이터에 대한 모델의 성능을 평가하기 위해, Training data로 사용했던 데이터를 모델의 평가에 활용하지 않는다.

## Test 세트가 오염되었다는 말은?
Test data에 Training data와 일치하거나, 매우 유사한 데이터들이 포함되어 Test data가 General한 상황에서의 성능 평가를 수행하지 못함을 말한다.

## Validation 세트가 따로 있는 이유는?
모델 학습 과정 중, Training data와 분리된 Validation data로 모델을 평가하여 그 결과를 학습에 반영하므로써 Training Data에 대한 Overfitting을 방지하는 효과가 있다.

## Dropout이란?
Dropout은 노드들의 연결을 무작위로 끊는 방식으로, 하나의 노드가 너무 큰 가중치를 가져 다른 노드들의 학습을 방해하는 현상을 억제한다. 이를 통해 모델의 일반화 성능을 높이고 Overfitting을 방지하는 효과를 얻을 수 있다. 

## Average Pooling과 Max Pooling의 차이점은?
Pooling은 Feature map에서 feature 수를 감소시키는 역할을 한다. Average Pooling은 kernel window에 해당하는 값들의 평균을 대표로, Max Pooling은 가장 큰 값을 대표로 feature 수를 감소시킨다. Max pooling은 kernel 영역 내에서 가장 두드러지는 값을 남기고, average는 영역 내의 모든 값을 고려하는 효과가 있다.

## Max Pooling의 장/단점은?
Max pooling의 장점은 데이터의 차원이 감소하여 신경망의 계산효율성이 좋아진다는 장점이 있다. 하지만 신경망이 깊어질수록 feature를 미세하게 보존할 수 없다는 단점이 있다.

## Fully Connected Layer의 기능은 무엇인가?
번역 그대로 완전 연결 된 계층이다. 모든 뉴런이 그 다음 층의 모든 뉴런과 연결된 상태를 말한다. 여기에 활성화 함수를 적용해서 이미지를 분류하는 기능을한다.

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

## Ground truth란?
딥러닝 관점에서 Ground truth란 학습하고자 하는 데이터의 원본 혹은 실제 값을 표현할 떄 사용된다.

## Occlusion이란?
Object tracking 분야에서 물체가 겹치거나 가려 인식을 못하는 경우를 말한다.

## self-supervised learning이란?
label이 없는 Untagged data를 기반으로 한 학습으로 자기 스스로 학습 데이터에 대한 분류를 수행하기 때문에 self가 붙는다. [관련링크](https://lifeisenjoyable.tistory.com/15)