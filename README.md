
# fault-detection

+ 머신러닝 관점에서 단층 해석
  > amplitude scaling
    : 탄성파 영상의 진폭을 정규화 하거나 표준화 하는 작업
    1. MaxAbs scaler
      : 진폭 절댓값의 최대값으로 전체 진폭을 scaling

![image](https://github.com/user-attachments/assets/cfa2dadd-6c4a-4830-ba91-aad450193cf6)   
    2. Min-max scaler (Standardization)
      : 진폭 최대/최솟값을 0과 1사이의 값으로 linear mapping
![image](https://github.com/user-attachments/assets/f3fdbb20-6ba1-4f28-ba00-36184f93bc2a)

    3. Standard scaler (Normalization)
      : 진폭의 평균과 표준편차를 이용하여 Gaussian 분포로 표준화
![image](https://github.com/user-attachments/assets/0a94b50c-f717-48c7-98c6-09a399518ffe)


    4. Symmetric Clipped Maxabs scaler
      :mode값을 기준으로 좌우 대칭이 되도록 trimming
![image](https://github.com/user-attachments/assets/6d8f2153-562d-4883-a672-3e4ebcabdb17)


    5. Normal score transfom
	    : 데이터 순위(rank)를 이용한 비정규 분포 자료의 표준 정규 분포 	형태로의 변환
![image](https://github.com/user-attachments/assets/453068e9-c5c1-4199-a1e5-5530c430efa9)



    
