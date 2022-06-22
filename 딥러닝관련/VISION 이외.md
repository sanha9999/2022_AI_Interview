## Attention이란?

RNN에 기반한 모델이 가지고 있는 문제는 기울기 소실(Vanishing Gradient)과 정보 손실 문제이다. 이로 인해 input으로 들어가는 문장이 길어지면 품질이 떨어지는 현상이 나타난다. 이를 해결하기 위해 등장한 방법이 바로 Attention이다. Attention mechanism은, 디코더에서 출력 단어를 예측하는 매 시점마다 인코더에서의 전체 입력 문장 중 해당 시점에서 예측해야 할 단어와 연관성있는 부분을 좀 더 집중해서 보는 것이다.

## Transformer란?

Transformer는 2017년 구글이 발표한 논문인 "Attention is all you need"에서 나온 모델로 seq2seq의 구조인 인코더-디코더를 따르면서도, 논문의 이름처럼 Attention으로만 구현한 모델이다. Transformer는 RNN을 사용하지 않고, 인코더-디코더 구조를 설계하였음에도 번역 성능에서 RNN을 능가하는 성능을 보여주었다.

## Collaborative filtering이란?

Collaborative filtering은 협업 필터링이라 하여 추천 시스템에서 가장 많이 사용되는 기술로, 유저-아이템 간 상호 작용 데이터를 활용하는 방법론이다. 예를 들면 "이 영화를 좋아했던 다른 사람들은 또 어떤 영화를 좋아해요?"

협업 필터링 뿐만 아니라 추천 모델에는 콘텐츠 기반 필터링(Content-based Filtering)이 있는데, [kakao Tech](https://tech.kakao.com/2021/10/18/collaborative-filtering/)에서 진짜 야무지게 설명해놓았다. 참고하면 좋다.