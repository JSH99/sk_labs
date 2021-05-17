import cv2
from PIL import Image
import argparse #ArgumentParser사용하기 위해

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy() #매개변수로 받은 frame을 copy함
    frameHeight=frameOpencvDnn.shape[0] #frameOpencvDnn의 전체 행의 갯수=>Height에 저장
    frameWidth=frameOpencvDnn.shape[1] ##frameOpencvDnn의 전체 열의 갯수=>Width에 저장

    #이미지 전처리
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob) #네트워크 입력설정 모델에 들어가는 input
    detections=net.forward()#정방향실행

    faceBoxes=[] #배열 선언
    #얼굴탐지
    for i in range(detections.shape[2]): #여러명이 있을수 있으므로 반복문
        confidence=detections[0,0,i,2]

        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)

            faceBoxes.append([x1,y1,x2,y2])#배열에

            #얼굴구간 네모 박스
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


parser=argparse.ArgumentParser() #인자값 받을수 있는 인스턴스 생성
parser.add_argument('--image') #입력받을 인자값 등록 :

args=parser.parse_args() #인자값 args에 저장

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"

#age_deploy는 텐서플로우, age_net은 카페// 예측모델 불러오기
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"

# 사전에 학습된 가중치 파일 불러오기(경험적 최적값)
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)

# 배열의 저장
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

#cv2.dnn_net 클래스 객체 생성
#훈련된 가중치를 저장하고 있는 이진파일이름: faceNodel,명시적 딥러닝 프레임워크이름:faceproto
faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)

video=cv2.VideoCapture(args.image if args.image else 0) #VideoCapture(0 또는 1)->실시간 frame을 받앙옴
padding=20
child=0
adult=0
old=0

endbreak=True
while cv2.waitKey(1)<0: #키 입력 대기시간
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg,faceBoxes=highlightFace(faceNet,frame) #함수 호출
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        #얼굴이미지 추출
        face=frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),
                    max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        # 연령 예측을 위한 이미지 변환 맟 전처리
        # 입력 영상을 블롭 객체로 만들어서 추론해야함( 모델파일이 어떻게 학습되었는지 파악하고 알맞게 지정)
        # 블롭객체=cv2.dnn.blobFromImage(입력영상,입력 영상의 픽셀에 곱할값, 입력 영상 각 채널에서 뺄 평균 값(경험적 학습값) ,r과 g의 값을 바꿀것인지)

        # 나이 예측
        ageNet.setInput(blob) #네트워크 입력 설정
        agePreds=ageNet.forward() #정방향 실행
        age=ageList[agePreds[0].argmax()]# 가장 높은 score값 선정
        print(f'age: {age[1:-1]} years') #문자열 만들기 위한 f'

        #cv2.puttext(이미지파일, 출력할 문자, ,크기와 글꼴, 0.8,(0,255,255) , 글짜 두께, 선 표시 방법)
        # 예측된 정보를 테두리 안에 입력
        cv2.putText(resultImg, f'{age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        #이미지 화면 출력(파일창 이름, 파일명)
        cv2.imshow("Detecting age", resultImg)

#예측값 count
        if f'{age}' == '(0-2)' or f'{age}' == '(4-6)' or f'{age}' == '(8-12)':
            print("age---------------어린이")
            child+=1
        elif f'{age}' == '(15-20)' or f'{age}' == '(25-32)' or f'{age}' == '(38-43)' or f'{age}' == '(48-53)':
            print("age---------------성인")
            adult+=1
        else:
            print("age---------------노년층")
            old+=1

#10번 이상 같은 예측값이 나왔을때
        if child>15 or adult>15 or old>15:
            if child>10:
                path1 = './어린이선택.png'
                path2 = './어린이메뉴.png'
                print("판독 결과: 어린이입니다")
                endbreak=False
                break

            elif adult>10:
                path1 = './성인선택.png'
                path2 = './성인메뉴.png'
                print("판독 결과: 성인입니다")
                endbreak=False
                break

            else:
                path1 = './노인선택.png'
                path2 = './노인메뉴.png'
                print("판독 결과: 노인입니다")
                endbreak=False
                break

    if(endbreak==False):
        break

#맞춤 화면 띄움

im1 = Image.open(path1)
im1.show()
im2 = Image.open(path2)
im2.show()

