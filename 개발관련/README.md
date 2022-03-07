# AI 개발관련 지식
## Gradient Descent란?
Gradient Descent, 즉 경사 하강 알고리즘은 Cost Function(비용 함수)의 값을 최소화하는 파라미터를 찾는 알고리즘이다. 기본적인 개념은 함수의 기울기를 구하여 기울기가 낮은 쪽으로 계속 이동시켜 최적값에 이를 때까지 반복하는 것이다. Gradient Descent의 약점은 현재 위치에서의 기울기를 사용하기 때문에 local minimum에 빠질 수 있다는 점이다. 그래서 추후에 모멘텀이라는 방식이 등장하게 된다.

## Loss Surface란?
모델을 훈련시킨다는 말은 비용 함수에서 파라미터를 업데이트하며 global minimum을 찾는 과정이다. Loss Surface란 global minimum을 찾아가는 과정을 시각화한 것으로, 모델을 이해하고 설계하는데 인사이트를 준다.

## Attention이란?
RNN에 기반한 모델이 가지고 있는 문제는 기울기 소실(Vanishing Gradient)과 정보 손실 문제이다. 이로 인해 input으로 들어가는 문장이 길어지면 품질이 떨어지는 현상이 나타난다. 이를 해결하기 위해 등장한 방법이 바로 Attention이다. Attention mechanism은, 디코더에서 출력 단어를 예측하는 매 시점마다 인코더에서의 전체 입력 문장 중 해당 시점에서 예측해야 할 단어와 연관성있는 부분을 좀 더 집중해서 보는 것이다.