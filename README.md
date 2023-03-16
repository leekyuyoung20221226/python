1차 프로젝트 예비 주제



1. 텍스트 데이터 + 주식데이터 분석을 해서 예측프로세스
     시계열과 자연어처리  (머신러닝 또는 딥러닝)     

2. 의료 : 데이터 확보. 예측.

3. 집값예측 -  집값 예측..  


코렙에서 matplotlib 한글 처리

코렙에서 matplotlib 한글
1. 다음을 설치
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf

2. 런타임 재시작
3. 
import matplotlib.pyplot as plt
plt.rc('font',family = 'NanumBarunGothic')
