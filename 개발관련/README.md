# AI 개발관련 지식
## Gradient Descent란?
Gradient Descent, 즉 경사 하강 알고리즘은 Cost Function(비용 함수)의 값을 최소화하는 파라미터를 찾는 알고리즘이다. 기본적인 개념은 함수의 기울기를 구하여 기울기가 낮은 쪽으로 계속 이동시켜 최적값에 이를 때까지 반복하는 것이다. Gradient Descent의 약점은 현재 위치에서의 기울기를 사용하기 때문에 local minimum에 빠질 수 있다는 점이다. 그래서 추후에 모멘텀이라는 방식이 등장하게 된다.

## Loss Surface란?
모델을 훈련시킨다는 말은 비용 함수에서 파라미터를 업데이트하며 global minimum을 찾는 과정이다. Loss Surface란 global minimum을 찾아가는 과정을 시각화한 것으로, 모델을 이해하고 설계하는데 인사이트를 준다.

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